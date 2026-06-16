import gc
from os import getcwd
from os.path import join
from datetime import timedelta
from time import time
import os
import glob
import subprocess
import nemo.collections.asr as nemo_asr
from models.ModelWrapper import ModelWrapper


class Parakeet_Hybrid(ModelWrapper):

    name = "parakeet-tdt-0.6b-v2"
    transcription = {}
    vtt = {}
    load_time = {}
    transcribe_time = {}

    def __init__(self, options=None):
        # call like Parakeet_Hybrid(options={"sliding_window": True, "context_size": [128, 128]})
        self.options = options or {}
        self._asr_model = None

    def load(self):
        start = time()
        self._asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/" + self.name)
        
        if self.options.get("sliding_window", False):
            context_size = self.options.get("context_size", [128, 128])
            
            if hasattr(self._asr_model, "change_attention_model"):
                self._asr_model.change_attention_model(
                    self_attention_model="rel_pos_local_attn", 
                    att_context_size=context_size
                )
                print(f"[{self.name}] Sliding window attention enabled with context size: {context_size}")
            else:
                print(f"[{self.name}] Warning: change_attention_model not available on this model instance.")

        end = time()
        self.load_time = str(timedelta(seconds=end - start))

    def unload(self):
        del self._asr_model
        gc.collect()

    def _split_audio(self, audio_file, output_dir):
        # split to 1440 sec, or 24 min

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_file,
                ],
                capture_output=True, text=True, check=True,
            )

            duration = float(result.stdout.strip())

            if duration <= 1440:
                return [audio_file]

        except Exception as e:
            print(f"[{self.name}] Failed to get audio duration: {e}")
            return []

        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        ext = os.path.splitext(audio_file)[1]

        chunk_pattern = os.path.join(
            output_dir,
            f"{base_name}_chunk_%03d{ext}"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            audio_file,
            "-f",
            "segment",
            "-segment_time", "1440", 
            "-c",
            "copy",
            chunk_pattern,
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"[{self.name}] ffmpeg segmentation failed: {e}")
            return []

        search_pattern = os.path.join(output_dir, f"{base_name}_chunk_*")
        return sorted(glob.glob(search_pattern))

    def _merge_segments(self, all_chunk_segments):
        merged_segments = []
        chunk_offset = 1440.0 # 24 minutes in seconds

        for chunk_idx, segments in enumerate(all_chunk_segments):
            offset = chunk_idx * chunk_offset
            for seg in segments:
                new_seg = seg.copy()
                if 'start' in new_seg:
                    new_seg['start'] = round(new_seg['start'] + offset, 2)
                if 'end' in new_seg:
                    new_seg['end'] = round(new_seg['end'] + offset, 2)
                if 'words' in new_seg:
                    new_seg['words'] = [
                        {**w, 'start': round(w['start'] + offset, 2), 'end': round(w['end'] + offset, 2)}
                        for w in new_seg['words']
                    ]
                merged_segments.append(new_seg)
                
        return merged_segments

    def transcribe(self, audio_name, audio_file, prompt=None, output_dir=os.getcwd()):
        start = time()

        # chunk audio
        chunks = self._split_audio(audio_file, output_dir)
        if not chunks:
            print(f"[{self.name}] Error: Failed to split audio file.")
            return

        all_texts = []
        all_chunk_segments = []

        # transcribe each chunk sequentially
        for chunk in chunks:
            output = self._asr_model.transcribe([chunk], timestamps=True)
            
            # empty returns
            if not output or not output[0]:
                print(f"[{self.name}] Warning: Chunk {chunk} returned no transcription.")
                continue

            result = output[0]

            # Extract text and segments based on object type
            if hasattr(result, "text"):
                chunk_text = result.text.strip()
                chunk_segs = result.timestamp.get("segment", []) if hasattr(result, "timestamp") else []
            else:
                chunk_text = str(result).strip()
                chunk_segs = []

            if chunk_text:
                all_texts.append(chunk_text)
            if chunk_segs:
                all_chunk_segments.append(chunk_segs)

        end = time()
        self.transcribe_time[audio_name] = str(timedelta(seconds=end - start))

        # concatenate text with a space separator
        self.transcription[audio_name] = " ".join(all_texts)

        # merge VTT segments with progressive time offsets
        merged_segments = self._merge_segments(all_chunk_segments)
        self.vtt[audio_name] = self.__generate_vtt(merged_segments)

        # save output
        self.__write_output(audio_name, self.transcription[audio_name], self.vtt[audio_name], output_dir)

        # cleanup temp chunk files
        for chunk in chunks:
            if chunk == audio_file:
                continue

            try:
                os.remove(chunk)
            except OSError:
                pass

    def __write_output(self, audio_name, text, vtt_text, output_dir):
        txt_path = join(output_dir, f"{audio_name}.txt")
        vtt_path = join(output_dir, f"{audio_name}.vtt")
        print(f"[{self.name}] Wrote VTT -> {vtt_path}")
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            with open(vtt_path, "w", encoding="utf-8") as f:
                f.write(vtt_text)
        except Exception as e:
            print(f"Failed to write outputs: {e}")

    def __generate_vtt(self, segments):
        if not segments:
            return "WEBVTT\n\n[No timestamp segments generated]"
            
        lines = ["WEBVTT\n"]
        for idx, seg in enumerate(segments):
            start = self.__format_timestamp(seg["start"])
            end = self.__format_timestamp(seg["end"])
            text = seg["segment"]
            lines.append(f"{idx+1}\n{start} --> {end}\n{text}\n")
        return "\n".join(lines)

    def __format_timestamp(self, seconds):
        millis = int((seconds % 1) * 1000)
        total_seconds = int(seconds)
        mins, secs = divmod(total_seconds, 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02}:{mins:02}:{secs:02}.{millis:03}"
