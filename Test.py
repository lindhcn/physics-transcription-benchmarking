from os import getcwd
from os.path import join
from inspect import getsource
from datetime import datetime
from shutil import rmtree
from helper_functions.prompt_functions import no_prompt
from helper_functions.test_transcribe_help import load_dataset, string_to_timedelta, compare, merge_dicts, summarize, make_output_folders
from create_test_summary.TestSummary import create_test_summary_html
from whisper.normalizers import EnglishTextNormalizer
import gc, json, platform, psutil, copy

# ==================== #
# ==== Test Class ==== #
# ==================== #

class Test():

    model_array = []
    prompt_function_array = []

    def __init__(self, model_array, prompt_function_array=[no_prompt], output_dir=getcwd()):
        self.model_array = model_array
        self.prompt_function_array = prompt_function_array
        self.normalizer = EnglishTextNormalizer()
        self.output_dir = output_dir
        self.most_recent_run_name = None
        self.most_recent_run_results = None
        self.results_folder = None
        self.__transcriptions_folder = None
        self.__temp_folder = None

    def run(self, run_name, dataset_path, run_num=1, save_transcription=False):
        self.most_recent_run_name = run_name

        # CREATING OUTPUT FOLDERS:
        
        self.results_folder, self.__transcriptions_folder, self.__temp_folder = make_output_folders(output_dir=self.output_dir, 
                                                                                                    run_name=run_name, 
                                                                                                    dirs_to_make=[True, save_transcription, True])

        # LOADING DATASET:
        
        dataset = load_dataset(dataset_path)
        if dataset == None:
            print("Invalid dataset path provided: '"+dataset_path+"'")
            return

        # GETTING SYSTEM/MEMORY INFORMATION:

        uname = platform.uname()
        mem = psutil.virtual_memory()

        # RUNNING TESTS:

        for model in self.model_array:

            # load model
            model.load()

            # get additional model attributes
            model_attributes = {}
            for key, value in model.__dict__.items():
                if key[0] != '_' and key != "name":
                    model_attributes.update({key: value})

            for prompt_function in self.prompt_function_array:

                current_model = {}
                test_results = {}
                test_summary = {}

                # create test_details dictionary
                test_details = {"model_info": {"class_name": model.__class__.__name__,
                                            "model_name": model.name,
                                            **model_attributes,
                                            "load_time": model.load_time},
                                "prompt_info": {"prompt_function_name": prompt_function.__name__,
                                                "prompt_function_code": getsource(prompt_function)},
                                "system_info": {"system": uname.system,
                                                "release": uname.release,
                                                "version": uname.version,
                                                "machine": uname.machine,
                                                "processor": uname.processor},
                                "cpu_info": {"physical_cores": psutil.cpu_count(logical=False),
                                            "total_cores": psutil.cpu_count(logical=True)},
                                "memory_info": {"total_memory": mem.total,
                                                "available_memory": mem.available,
                                                "used_memory": mem.used}}
                # TODO: add GPU details?

                # add test_details dict to current model
                current_model.update({"test_details": test_details})

                for test_case in dataset:

                    local_test_results = {}
                    local_summary = {}
                    audio_name = test_case["audio_name"]
                    audio_file = test_case["audio_file"]
                    transcript_file = test_case["transcript_file"]

                    for i in range(run_num):

                        local_rerun_test_results = {}
                    
                        # creating prompt
                        prompt = prompt_function(test_case["audio_info"])

                        # transcribing model
                        model.transcribe(audio_name, join(dataset_path, "test_data", audio_file), prompt, self.__transcriptions_folder)

                        if save_transcription:
                            transcription = model.transcription[audio_name]
                            with open(join(self.__transcriptions_folder, model.name + "_" + prompt_function.__name__ + "_" + audio_name + ".txt"), "w") as f:
                                f.write(transcription)
                            with open(join(self.__transcriptions_folder, model.name + "_" + prompt_function.__name__ + "_" + audio_name + "-normalized.txt"), "w") as f:
                                f.write(self.normalizer(transcription))
        
                        # adding current date and transcribe time to result dict
                        local_rerun_test_results.update({"start_datetime": datetime.now().strftime("%D, %H:%M:%S")})
                        if model.transcribe_time[audio_name]:
                            transcribe_time = model.transcribe_time[audio_name]

                            # add to current test dict
                            local_rerun_test_results.update({"transcribe_time": transcribe_time})

                            # convert string to timedelta
                            transcribe_time = string_to_timedelta(transcribe_time)

                        # evaluating transcription
                        with open(join(dataset_path, "test_data", transcript_file), "r") as f:
                            reference = f.read()
                        accuracy_data = compare(self.normalizer(reference), self.normalizer(model.transcription[audio_name]))
                        
                        # updating dictionaries
                        run_data = {"transcribe_time": transcribe_time, **accuracy_data}
                        local_summary = merge_dicts(local_summary, run_data)
                        test_summary = merge_dicts(test_summary, run_data)
                        local_rerun_test_results.update(accuracy_data)
                        local_test_results.update({"run_"+str(i): local_rerun_test_results})

                        # freeing memory
                        del local_rerun_test_results
                        del prompt
                        del reference
                        del accuracy_data

                    # adding local summary to local test results dictionary
                    local_test_results.update({"summary": summarize(local_summary)})

                    # updating test result dictionary
                    test_results.update({test_case["audio_name"]: local_test_results})

                    # freeing memory
                    del local_test_results
                    del local_summary
                    del audio_name
                    del audio_file
                    del transcript_file
                    del test_case
                    gc.collect()

                # finalizing test summary dictionary
                test_summary = {"transcriptions_per_audio": run_num, **summarize(test_summary)}

                # adding test_results and test_summary to model dictionary 
                current_model.update({"test_results": test_results, "test_summary": test_summary})

                # adding model dict to run results array:
                self.most_recent_run_results = current_model

                # creating json object for model
                json_obj = json.dumps(current_model, indent=4)

                # writing json object to file
                with open(join(self.results_folder, model.name + "_" + prompt_function.__name__ + "_results.json"), "w") as f:
                    f.write(json_obj)

                # freeing memory
                del current_model
                del test_results
                del test_summary
                del test_details
                del json_obj
                del prompt_function
                gc.collect()

            # freeing memory
            model.unload()
            del model_attributes
            del model
            gc.collect()

        # freeing memory
        rmtree(self.__temp_folder)
        del self.__transcriptions_folder
        del self.__temp_folder
        del dataset
        del uname
        del mem
        gc.collect()

    def addModel(self, new_model):
        self.model_array.append(new_model)

    def removeModel(self, existing_model_name):
        for model in self.model_array:
            if model.name == existing_model_name:
                self.model_array.remove(model)

    def addPromptFunction(self, new_prompt_func):
        self.prompt_function_array.append(new_prompt_func)

    def removePromptFunction(self, existing_prompt_func_name):
        for prompt_function in self.prompt_function_array:
            if prompt_function.__name__ == existing_prompt_func_name:
                self.prompt_function_array.remove(prompt_function)

    def createSummaryHTML(self, html_filename=None):
        if self.results_folder == None:
            print("Please use run() before creating summary HTML.")
            return

        if html_filename == None:
            html_filename = self.most_recent_run_name
        create_test_summary_html(results_folder=self.results_folder,
                                filename=html_filename)

    def free(self):
        del self.model_array
        del self.prompt_function_array
        del self.normalizer
        del self.most_recent_run_name
        del self.most_recent_run_results
        del self.results_folder
        gc.collect()

# ================================= #
# ==== AddToExistingTest Class ==== #
# ================================= #

'''
Notes: This class will update the given json test output file.
       If no output file name is provided, it will likely 
       overwrite the provided test file.

       The 'test_details' section of the provided test file will 
       be overwritten with data from the provided transcription 
       model (meaning information in the 'test_details' section 
       of the previous run will be lost).
'''
    
class AddToExistingTest():

    def __init__(self, existing_test_json, dataset_path, model, prompt_function=no_prompt, output_dir=getcwd()):
        self.dataset_path = dataset_path
        self.model = model
        self.prompt_function = prompt_function
        self.normalizer = EnglishTextNormalizer()
        self.output_dir = output_dir
        self.most_recent_run = None
        self.results_folder = None
        self.__temp_folder = None

        # LOADING PROVIDED MODEL:

        model.load()

        # get additional model attributes
        model_attributes = {}
        for key, value in model.__dict__.items():
            if key[0] != '_' and key != "name":
                model_attributes.update({key: value})

        # LOADING EXISTING TEST DATA:

        existing_test_file = open(existing_test_json)
        existing_test_obj = json.load(existing_test_file)
        existing_test_file.close()

        existing_test_details = existing_test_obj["test_details"]
        self.existing_test_results = existing_test_obj["test_results"]

        # CONFIRMING IF PROPER MODEL AND PROMPT FUNCTION:

        self.existing_model_info = existing_test_details["model_info"]
        self.provided_model_info = {"class_name": model.__class__.__name__,
                               "model_name": model.name,
                               **model_attributes,
                               "load_time": model.load_time}
        existing_prompt_info = existing_test_details["prompt_info"]
        provided_prompt_info = {"prompt_function_name": prompt_function.__name__,
                                "prompt_function_code": getsource(prompt_function)}

        # reporting discrepancies between models/prompts
        for parameter in self.existing_model_info:
            if parameter != "model_name" and parameter != "load_time":
                if parameter not in self.provided_model_info or self.existing_model_info[parameter] != self.provided_model_info[parameter]:
                    self.__discrepency_error(parameter, self.existing_model_info, self.provided_model_info)
                    
        for parameter in existing_prompt_info:
            if parameter not in provided_prompt_info or existing_prompt_info[parameter] != provided_prompt_info[parameter]:
                self.__discrepency_error(parameter, existing_prompt_info, provided_prompt_info)

        # freeing memory
        del model_attributes
        del existing_test_obj
        del existing_test_details
        del existing_prompt_info
        del provided_prompt_info
        gc.collect()

    def run(self, run_name, run_num=1, output_file_name=None):
        self.most_recent_run = run_name

        #  CREATING OUTPUT FOLDERS:

        self.results_folder, _, self.__temp_folder = make_output_folders(output_dir=self.output_dir, 
                                                                        run_name=run_name, 
                                                                        dirs_to_make=[True, False, True])

        # LOADING DATASET:

        dataset = load_dataset(self.dataset_path)
        if dataset == None:
            print("Invalid dataset path provided: '"+self.dataset_path+"'")
            return

        # GETTING PROVIDED MODEL STATS:

        uname = platform.uname()
        mem = psutil.virtual_memory()
        self.provided_test_details = {"model_info": self.provided_model_info,
                        "prompt_info": {"prompt_function_name": self.prompt_function.__name__,
                                        "prompt_function_code": getsource(self.prompt_function)},
                        "system_info": {"system": uname.system,
                                        "release": uname.release,
                                        "version": uname.version,
                                        "machine": uname.machine,
                                        "processor": uname.processor},
                        "cpu_info": {"physical_cores": psutil.cpu_count(logical=False),
                                    "total_cores": psutil.cpu_count(logical=True)},
                        "memory_info": {"total_memory": mem.total,
                                        "available_memory": mem.available,
                                        "used_memory": mem.used}}
    
        # RUNNING TESTS:

        current_model = {"test_details": self.provided_test_details}
        test_results = {}
        test_summary = {}
        
        for test_case in dataset:

            local_test_results = {}
            local_summary = {}
            num_prev_runs = 0
            audio_name = test_case["audio_name"]
            audio_file = test_case["audio_file"]
            transcript_file = test_case["transcript_file"]

            # updating starting param values if this audio already has existing results
            if audio_name in self.existing_test_results:

                existing_audio_name = self.existing_test_results[audio_name]
                num_prev_runs = len(existing_audio_name)-1

                for run in existing_audio_name:

                    if run != "summary":
                        # update local test results
                        local_test_results.update({run: existing_audio_name[run]}) 

                        # deep copy and alter existing test data as needed
                        run_copy = copy.deepcopy(existing_audio_name[run])
                        del run_copy["start_datetime"]
                        run_copy["transcribe_time"] = string_to_timedelta(run_copy["transcribe_time"])

                        # add altered existing test data to summary dicts
                        local_summary = merge_dicts(local_summary, run_copy)
                        test_summary = merge_dicts(test_summary, run_copy)
                          
            for i in range(num_prev_runs, run_num + num_prev_runs):

                local_rerun_test_results = {}
            
                # creating prompt
                prompt = self.prompt_function(test_case["audio_info"])

                # transcribing model
                self.model.transcribe(audio_name, join(self.dataset_path, "test_data", audio_file), prompt, self.__transcriptions_folder)

                # adding current date and transcribe time to result dict
                local_rerun_test_results.update({"start_datetime": datetime.now().strftime("%D, %H:%M:%S")})
                
                if self.model.transcribe_time[audio_name]:
                    current_transcribe_time = self.model.transcribe_time[audio_name]

                    # add to current test dict
                    local_rerun_test_results.update({"transcribe_time": current_transcribe_time})

                    # convert string to timedelta and add to array
                    transcribe_time = string_to_timedelta(current_transcribe_time)

                # evaluating transcription
                with open(join(self.dataset_path, "test_data", transcript_file), "r") as f:
                    reference = f.read()
                accuracy_data = compare(self.normalizer(reference), self.normalizer(self.model.transcription[audio_name]))
                
                # updating dictionaries
                run_data = {"transcribe_time": transcribe_time, **accuracy_data}
                local_summary = merge_dicts(local_summary, run_data)
                test_summary = merge_dicts(test_summary, run_data)
                local_rerun_test_results.update(accuracy_data)
                local_test_results.update({"run_"+str(i): local_rerun_test_results})

                # freeing memory
                del local_rerun_test_results
                del prompt
                del reference
                del accuracy_data

            # adding local summary to local test results dictionary
            local_test_results.update({"summary": summarize(local_summary)})

            # updating test result dictionary
            test_results.update({test_case["audio_name"]: local_test_results})
            
            # freeing memory
            del local_test_results
            del local_summary
            del audio_name
            del audio_file
            del transcript_file
            del test_case
            gc.collect()

        # finalizing test summary dictionary
        test_summary = {"transcriptions_per_audio": run_num + num_prev_runs, **summarize(test_summary)}

        # updating json
        current_model.update({"test_results": test_results, "test_summary": test_summary})

        # writing json object to file
        json_obj = json.dumps(current_model, indent=4)
        if output_file_name != None:
            with open(join(self.results_folder, output_file_name), "w") as f:
                f.write(json_obj)
        else:
            with open(join(self.results_folder, self.existing_model_info["model_name"] + "_" + self.prompt_function.__name__ + "_results.json"), "w") as f:
                f.write(json_obj)

        # freeing memory
        rmtree(self.__temp_folder)
        del self.__temp_folder
        del dataset
        del uname
        del mem
        gc.collect()

    def free(self):
        # freeing model
        self.model.unload()
        self.dataset_path
        del self.model
        del self.prompt_function
        del self.normalizer
        del self.results_folder
        gc.collect()

    def __discrepency_error(self, parameter, existing_model_info, provided_model_info, isModel):
        model.unload()
        del model
        gc.collect()
        if isModel:
            raise Exception("Model parameter, '"+parameter+"', between test model (model used to create existing test) and provided model is different.\n\
                            Test model value: "+existing_model_info[parameter]+"\n\
                            Provided model value: "+provided_model_info[parameter])
        else:
            raise Exception("Prompt parameter, '"+parameter+"', between test prompt (prompt used to create existing test) and provided prompt is different.\n\
                            Test prompt value: "+existing_model_info[parameter]+"\n\
                            Provided prompt value: "+provided_model_info[parameter])
            
