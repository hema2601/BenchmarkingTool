import json
import subprocess
from enum import Enum
import time
import os
from datetime import datetime
import itertools
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import math
import tomli
import scipy.stats as stats


class min_max_avg:
    minimum = None
    maximum = None
    average = None
    count = None

    def __repr__(self):
        return "["+str(self.average)+", "+str(self.minimum)+", "+str(self.maximum)+"]"

    def __init__(self, average=0, minimum=sys.maxsize, maximum=0, count=0):
        self.count = count
        self.maximum = maximum
        self.average  = average
        self.minimum = minimum

    def compute_next(self, value):

        self.count += 1
        self.average = self.average + ((value - self.average) / self.count)

        if(value > self.maximum):
            self.maximum = value
        if(value < self.minimum):
            self.minimum = value

    def combine_mma(self, mma):
        if self.count !=  mma.count:
            print("This is probably an error...")
            return

        self.average = (self.average + mma.average) / 2

        if(mma.maximum > self.maximum):
            self.maximum = mma.maximum
        if(mma.minimum < self.minimum):
            self.minimum = mma.minimum

    def return_dict(self):
        ret_dict = dict()
        ret_dict["avg"] = self.average
        ret_dict["min"] = self.minimum
        ret_dict["max"] = self.maximum
        ret_dict["cnt"] = self.count

        return ret_dict



def runningAvg(avg, val, cnt):
    return float(avg) + ((float(val) - float(avg)) / float(cnt))




#TODO: Change the structure of this thing
class JsonAnalyzer:
    root_filename = None
    raw_file = None
    date = None
    avg_file = None
    filtered_avg_file = None

    def __init__(self, file_name="json.txt"):
        self.root_filename = file_name

        split = os.path.split(file_name)


        print(split)

        self.raw_file = open(split[0]+"/raw_"+split[1], 'w+', encoding="utf-8")
        

    def add(self, string):
        self.raw_file.write(string)

    def average(self):
        self.raw_file.seek(0)
        json_dict = json.load(self.raw_file)

        average_dict = dict()
        average_dict["Experiments"] = []

        for i, exp in enumerate(json_dict["Experiments"]):
            print(f"Experiment #"+str(i))
            average_dict["Experiments"].append(exp)
            average_dict["Experiments"][i]["Averages"] = collapse_experiment(exp)
            del average_dict["Experiments"][i]["Run Count"]
            del average_dict["Experiments"][i]["Runs"]

        split = os.path.split(self.root_filename)


        self.avg_file = open(split[0]+"/avg_"+split[1], "w+")

        self.avg_file.write(json.dumps(average_dict))


    def filtered_average(self):


        threshold = 3 #TODO: Get this from the user?

        self.raw_file.seek(0)
        json_dict = json.load(self.raw_file)


        #Determine how many averages need to be calculated
        value_names = list()

        for out in json_dict["Experiments"][0]["Runs"][0]["Outputs"]:
            for key in out.keys():
                value_names.append(key)

        print(value_names)

        

        #Create properly formatted dictionary 
        filtered_avg_dict = dict()
        filtered_avg_dict["Experiments"] = []
        
        #Calculate filtered average for every experiment
        for i, exp in enumerate(json_dict["Experiments"]):

            #Format Experiment Object in Dict
            filtered_avg_dict["Experiments"].append(exp)
            filtered_avg_dict["Experiments"][i]["Filtered Avg"] = dict() 

            #Collect all necessary data
            data_points = {key: [] for key in value_names}
            for run in exp["Runs"]:
                for out in run["Outputs"]:
                    for key in out.keys():
                        data_points[key].append(out[key])

            #Filter out outliers and compute min/max/avg
            for key in data_points.keys():
                zscore = stats.zscore(data_points[key])

                filtered = [data_points[key][i] for i in range(len(data_points[key])) if abs(zscore[i]) < threshold]
                data_points[key] = filtered

                filtered_avg_dict["Experiments"][i]["Filtered Avg"]["avg"] = float(np.mean(data_points[key]))
                filtered_avg_dict["Experiments"][i]["Filtered Avg"]["max"] = int(np.max(data_points[key]))
                filtered_avg_dict["Experiments"][i]["Filtered Avg"]["min"] = int(np.min(data_points[key]))
                filtered_avg_dict["Experiments"][i]["Filtered Avg"]["count"] = len(data_points[key])

            #remove unnecessary kv pairs
            del filtered_avg_dict["Experiments"][i]["Run Count"]
            del filtered_avg_dict["Experiments"][i]["Runs"]
       

        #Write to File
        split = os.path.split(self.root_filename)
        self.filtered_avg_file = open(split[0]+"/filtered_avg_"+split[1], "w+")
        self.filtered_avg_file.write(json.dumps(filtered_avg_dict))



    def cdf(self):
       
        #Create Folder for CDF graphs
        dirpath = os.path.split(self.root_filename)[0] + "/CDF_Graphs/"
        os.mkdir(dirpath)
        
        #TODO: Get cdf options from config (Maybe? Pending Design Choice...)
        names=["Marshalling", "Unmarshalling"]
        #######################################

        #Retrieve raw data from json
        self.raw_file.seek(0)

        json_dict = json.load(self.raw_file)

        


        #Create a data list for every experiment
        graphs = dict()
        for x in range(json_dict["Experiment Count"]):
            graphs[x] = list([] for _ in range(len(names)))


        #Read data points from the dictionary
        for exp in json_dict["Experiments"]:
            for run in exp["Runs"]:
                for out in run["Outputs"]:
                    for i in range(len(names)):
                        if names[i] in out.keys():
                            graphs[exp["Experiment Index"]][i].append(out[names[i]])
   

        #For every selected experiment, create the requested CDF
        for exp_idx, data in graphs.items():
            #print(data)
            for i in range(len(names)):
                #Credit: GeeksForGeeks https://www.geeksforgeeks.org/how-to-calculate-and-plot-a-cumulative-distribution-function-with-matplotlib-in-python/
                count, bins_count = np.histogram(data[i], bins=100) #TODO: Make number of bins dynamic

                pdf = count / sum(count)

                cdf = np.cumsum(pdf)

                plt.clf()

                plt.plot(bins_count[1:], cdf, label="CDF")
                plt.legend()
    
                plt.savefig(dirpath + "CDF_Experiment#" + str(exp_idx) + "_" + names[i] + ".png")



    def generate_excel(self, parameter, values, source_file="avg", excel_name=None):
        
        #Create Excel File
        dirpath = os.path.split(self.root_filename)[0] + "/Excel_Files/"
        
        if os.path.isdir(dirpath) == False:
            os.mkdir(dirpath)

        if excel_name == None:
            excel_name = parameter + (("_" + value) for value in values) #TODO find better standard naming

        excel = open(dirpath + excel_name, "w+")

        #Read data from file
        json_source = None
        if source_file == "raw":
            print("Fatal Error: Do not support raw data formatting yet") #TODO: Fix this
            return
            json_source = self.raw_file
        elif source_file == "avg":
            json_source = self.avg_file
        else:
            print("Fatal Error: Unknown File Type")
            return

        json_source.seek(0)
        json_dict = json.load(json_source)
        

        #Verify Existing Restrains (So far, this function only supports generating excel files for benchmarks that have one varying variable)
        paras = []
        for exp in json_dict["Experiments"]:
            if exp["Parameters"][parameter] in paras:
                print("Fatal Error: generate_excel() not supporte for this type of benchmark")
                return
            else:
                paras.append(exp["Parameters"][parameter])

        #Sort Experiments based on para values
        exps = [x for _, x in sorted(zip(paras, json_dict["Experiments"]))]




        #Write each table
       
        
        #Header

        headers = list()
        headers.append(list())
        headers[0].append(parameter)

        print(len(headers))
        
        for tup in values:
            for val in tup:
                
                if (len(headers) - 1) < (tup.index(val)):
                    headers.append([" " for _ in range(len(headers[tup.index(val) - 1]) - 1)])
                    
                headers[tup.index(val)].append(val)

        for line in headers:
            curr = line[0]
            for i in range(len(line[1:])):
                if line[i+1] == curr:
                    line[i+1] = " "
                else:   
                    curr = line[i+1]
            excel.write(''.join((elem + ", ") for elem in line)[:-2] + "\n")


        """
        next_string = parameter + (''.join((", " + value[0]) for value in values)) #TODO Fix this ugly workaround
        
        excel.write(next_string + "\n")
        """

        #Data
        for exp in exps:
            next_string = str(exp["Parameters"][parameter])
            for tup in values:
                next_string += ", " + str(access_json(exp["Averages"], tup))

            excel.write(next_string + "\n")

        excel.close()
            


def access_json(json_dict, access_tuple):
    curr = json_dict
    for x in access_tuple:
        curr = curr[x]

    return curr





class DataObject:
    name = None
    data_type = None
    value = None
    
    def __init__(self, name, data_type, value = 0):
        self.name = name
        self.data_type = data_type
        self.value = value

    def json_string(self):
        if self.data_type == 0:
            return ("\"" + str(self.name) + "\" : " + str(self.value) )

        return ("\"" + str(self.name) + "\" : { \"Type\" : " + str(self.data_type) + ", \"Value\" : " + str(self.value) + " }")

    def set_value(self, value):
        self.value = value

class Command:

    command = None

    def __init__(self, binary, para_list):

        self.command = [binary]

        for para in para_list.parameters:
            self.command.append(para.format_opt())

    def get_command(self):
        return self.command

    def to_file(self, filename):
        self.command.append(">")
        self.command.append(filename)


class Parameter:
    name = None
    value = None
    curr_value = None
    idx = 0
    flag = None
    has_value = None
    ranged = None
    related = None



    def __init__(self, name, flag=None, values=None):
        self.name = name
        self.value = values
        self.flag = flag

        self.has_value = True if values != None else False 
        self.ranged = isinstance(values, list)

        if self.ranged == False:
            self.value = [self.value]

        related = False
        curr_value = self.value[self.idx]
    
    def __str__(self):
        if self.has_value == True:
            return self.name + " : " + str(self.curr_value)
        else:
            return self.name + " : True"
    def __repr__(self):
        if self.has_value == True:
            return self.name + " : " + str(self.curr_value)
        else:
            return self.name + " : True"

    def format_opt(self):
        if self.has_value == True:
            if self.flag is not None:
                return "-" + self.flag + " " + str(self.curr_value)
            else:
                return str(self.curr_value)
        else:
            if self.flag is not None:
                return "-" + self.flag 
            else:
                return ""

    def set_value(self, value):
        try:
            self.idx = self.value.index(value)
        except ValueError:
            print(f"{self.name}:{value}Requested Value not in Range")
            self.idx = 0

        if self.related == True:
            self.curr_value = value
        else:
            self.curr_value = self.value[self.idx] 


    def json_string(self):
        if self.has_value == True:
            if type(self.curr_value) == str:
                return "\"" + self.name + "\" : \"" + str(self.curr_value) + "\""
            else:
                return "\"" + self.name + "\" : " + str(self.curr_value)
        else:
            return "\"" + self.name + "\" : \"True\""

    def has_range(self):
        return self.ranged 

class ParameterList:
    parameters = None
    count = None
    relations = None
    def __init__(self):
        self.parameters = []
        self.count = 0
        self.relations = []

    def add_para(self, para):
        self.parameters.append(para)
        self.count += 1
    
    def set_relation(self, para1, para2):
        if para1 >= self.count or para2 >= self.count:
            print("Parameter Index not in Range")
            return
        self.relations.append((para1, para2))
        self.parameters[para1].related = True

    def fill_in_static_paras(self, tup):
        
        ranged_paras_idx = [i for i, para in enumerate(self.parameters) if para.has_range()]
    
        values = []
        ranged_idx = 0

        for i in range(self.count):
            if i in ranged_paras_idx:
                values.append(tup[ranged_idx])
                ranged_idx += 1
            elif self.parameters[i].has_value == True:
                values.append(self.parameters[i].value[0])
            else:
                values.append(None)

        return tuple(values)

    def complete_para_combs(self, combs):
    
        completed_combs = []

        for each in combs:
            completed_combs.append(self.fill_in_static_paras(each))

        return completed_combs

    def resolve_relations(self, combs):
        
        if len(self.relations) == 0:
            return combs

        for i in range(len(combs)):
            temp = list(combs[i])
            for relation in self.relations:
                temp[relation[0]] *= temp[relation[1]]
                temp[relation[0]] = int(temp[relation[0]])
            combs[i] = tuple(temp)

        return combs
                


    def get_para_combinations(self):

        #Resolve Ranged Parameters
        combs = self.resolve_ranged()

        #Add Static Parameters
        combs = self.complete_para_combs(combs)

        #Resolve Relations
        combs = self.resolve_relations(combs) 

        #Remove Duplicates
        combs = list(set(combs))

        return combs


    def resolve_ranged(self):
        ranged_paras = [para.value for para in self.parameters if para.has_range()]
        #print(ranged_paras)

        if len(ranged_paras) == 0:
            return None

        comb = ranged_paras[0]

        is_first = True
        for para in ranged_paras[1:]:
            if is_first == True:
                comb = list(itertools.product(comb, para))
                is_first = False
            else:
                comb = [(*flatten, rest) for flatten, rest in list(itertools.product(comb, para))]

        if is_first == True:
            ret = []
            for x in comb:
                ret.append((x, ))
            return ret

        return comb

    def assign_paras(self, values):
        for i, para in enumerate(self.parameters):
            if para.has_value == True:
                para.set_value(values[i])



class Run:
    #idx = None
    #command = None
    #analyzer = None
    #thread_count_mode = None
    #output_counter = 0

    def __init__(self, idx, command, analyzer, thread_count_mode=0):
        self.idx = idx
        self.command = command
        self.analyzer = analyzer
        self.thread_count_mode = thread_count_mode
        self.output_counter = 0

    def count_threads(self, proc):

        do = DataObject("Thread Count", 0) if self.thread_count_mode == 1 else DataObject("Thread Count", 2)
        tc = [0]
        while(proc.poll() == None):
        
            p2 = subprocess.run('/bin/ps --no-headers -o thcount '+ str(proc.pid), shell=True, stdout=subprocess.PIPE)
            out = int(p2.stdout.decode("utf-8"))
            if self.thread_count_mode == 2:
                tc.append(out)
            else:
                tc[0] = out if out > tc[0] else tc[0]
            time.sleep(0.1)
           
        if self.thread_count_mode == 1:
            do.set_value(tc[0])
        elif self.thread_count_mode == 2:
            do.set_value(tc[1:])

        self.analyzer.add( "{ " + do.json_string() + " }")
        self.output_counter += 1

    def run(self):
        self.print_prologue()
       
        f = open("output.txt", "w+")

        p = subprocess.Popen(self.command.get_command(), stdout=f)

        if self.thread_count_mode != 0:
            self.count_threads(p)
            self.analyzer.add(", ")

        p.wait()
    
        f.close()
        f = open("output.txt", "r")
                
        self.analyzer.add(f.read())

        f.close()

        self.output_counter += 1
        
        self.print_epilogue()

    def print_prologue(self):
        self.analyzer.add("{ \"Run Index\" : " + str(self.idx) + ", \"Outputs\" : [ ")

    def print_epilogue(self):
        self.analyzer.add("], \"Output Count\" : " + str(self.output_counter) + " }")

class ExperimentObj:
    idx = None
    parameters = None
    command = None
    binary = None
    analyzer = None
    run_counter = 0
    thread_count_mode = None 
    runs = None

    def __init__(self, analyzer, idx, binary, parameters, runs, thread_count_mode=0):
        self.analyzer = analyzer
        self.idx = idx
        self.parameters = parameters
        self.binary = binary
        self.command = Command(binary, parameters)
        self.thread_count_mode = thread_count_mode
        self.runs = runs
    
    def __str__(self):
        string = f"----\nExperiment #{self.idx}\nParameter List:\n"
        for para in self.parameters.parameters:
            string += str(para) + "\n"

        string += "\n\n"

        return string

    def run_experiment(self):
        self.print_prologue()
        is_first = True

        for i in range(self.runs):
            if is_first == False:
                self.analyzer.add(", ")
            is_first = False
            run = Run(self.run_counter, self.command, self.analyzer, self.thread_count_mode)
            self.run_counter += 1;
            run.run()



        self.print_epilogue()

    def print_prologue(self, print_no_val=False):
        self.analyzer.add("{ \"Experiment Index\" : " + str(self.idx) + ", \"Parameters\" : { ")

        not_first = False
        for i, para in enumerate(self.parameters.parameters):
            if para.has_value == True:
                if not_first == True:
                    self.analyzer.add(", ")
                else:
                    not_first = True
                    
                self.analyzer.add(para.json_string())
            elif print_no_val == True:
                if not_first == True:
                    self.analyzer.add(", ")
                else:
                    not_first = True
                self.analyzer.add(para.json_string())

        self.analyzer.add(" }, \"Runs\" : [ ")


    def print_epilogue(self):
        
        self.analyzer.add(" ], \"Run Count\" : " + str(self.run_counter) + " }")

class ConfigObj:
    """
    A class that abstracts the access to the parsed TOML dictionary.
    It translates complex accesses to the dictionary into simple funciton calls.
    It also facilitates easy implementation of future changes in the config's format.

    ...

    Attributes
    ----------
    conf_dict : dictionary
        the dictionary representing the parsed config file

    Methods
    -------
    get_name()
        Returns the name of the benchmark as defined in the config file
    get_runs()
        Returns the number of executions per parameter combination as defined in the config file
    get_threading()
        Returns the mode for thread couting as defined in the config file
    get_bin()
        Returns the path to the executable as defined in the config file
    get_paras()
        Returns a parameter list assembled from the values given in the config file

    """
    conf_dict = None


    def __init__(self, name):
        fp = open(name, "rb")
        self.conf_dict = tomli.load(fp)
        fp.close()

    def get_name(self):
        return self.conf_dict["BASICS"]["BENCHMARK_NAME"]
    def get_runs(self):
        return self.conf_dict["BASICS"]["RUNS"]
    def get_threading(self):
        return self.conf_dict["BASICS"]["THREAD_COUNTING"]
    def get_bin(self):
        return self.conf_dict["BASICS"]["PATH_TO_BIN"]
    def get_paras(self):
        #Create Parameter List
        parameters = ParameterList()
        for name, flag, values in zip(self.conf_dict["PARAMETERS"]["NAMES"], 
                                   self.conf_dict["PARAMETERS"]["SPECIFIERS"], 
                                   self.conf_dict["PARAMETERS"]["VALUES"]) :
            parameters.add_para(Parameter(name, 
                                          flag if flag != "" else None, 
                                          values if values != "" else None))
        return parameters 

class BenchmarkObj:
    date = None
    title = None
    analyzer = None
    experiment_count = 0
    binary = None
    parameters = None
    runs = None
    threading = None
    dir_path = None


    def __init__(self, title, binary, parameters, filename="json.txt", runs = 1, threading = 0):
        self.binary = binary
        self.title = title
        self.date = datetime.now()
        self.date_str = self.date.strftime("%Y-%m-%d_%H:%M")
        self.parameters = parameters
        self.runs = runs
        self.threading = threading

        self.dir_path = "./"+self.title+"_"+self.date_str+"/"

        os.mkdir(self.dir_path)
        self.analyzer = JsonAnalyzer(self.dir_path+filename)

        
    @classmethod
    def from_config(cls, config):
            

        return cls(config.get_name(),
                   config.get_bin(),
                   config.get_paras(),
                   "json.txt",
                   config.get_runs(),
                   config.get_threading())
        

    def run_benchmark(self):
        self.print_prologue()

        run_paras = self.parameters.get_para_combinations()    

        is_first = True
        for para_set in run_paras:
            if is_first == False:
                self.analyzer.add(", ")
            is_first = False
            self.parameters.assign_paras(para_set)
            experiment = ExperimentObj(self.analyzer, self.experiment_count, self.binary, self.parameters, self.runs, self.threading)
            print(experiment)
            self.experiment_count += 1
            experiment.run_experiment()




        self.print_epilogue()     
    def print_prologue(self):
        self.analyzer.add("{ \"Title\" : \"" + self.title + "\", \"Date\" : \"" + self.date_str + "\", \"Experiments\" : [ ")
    def print_epilogue(self):
        self.analyzer.add("], \"Experiment Count\" : " + str(self.experiment_count) + "}")

    def finalize(self):
        self.analyzer.raw_file.close()

    def run_post_processing(self):

        print("Begin Post Processing")

        #self.analyzer.average()
        self.analyzer.filtered_average()
        #self.analyzer.generate_excel("Values", [("Initialization", "avg"), ("RPC Header", "avg"), ("Marshalling", "avg"), ("Sending", "avg"), ("Polling", "avg"), ("Receiving", "avg"), ("Unmarshalling", "avg")], "avg", "averages_excel.txt")
        #self.analyzer.generate_excel("Values", [("Marshalling", "avg"), ("Marshalling", "min"), ("Marshalling", "max"), ("Unmarshalling", "avg"), ("Unmarshalling", "min"), ("Unmarshalling", "max")], "avg", "marshalling_unmarshalling_excel.txt")
        #self.analyzer.cdf()


def plot_cdf(fp, name):
    fp.seek(0)

    json_dict = json.load(fp)

    graphs = dict()

    #Create a data list for every experiment
    for x in json_dict["Experiment Count"]:
        graphs[x] = list()

    for exp in json_dict["Experiments"]:
        for run in exp["Runs"]:
            for out in run["Outputs"]:
                if name in out.keys():
                    graphs[exp["Experiment Index"]].append(out[name])

    print("Data Points for Exp 1:")
    print(graphs[1])


"""
    upper_limit = 0

    for exp in json_dict["Experiments"]:
        graphs[exp["Parameters"][main.name]].append(exp)
        if exp["Averages"][name]["avg"] > upper_limit:
            upper_limit = exp["Averages"][name]["avg"]
  
    upper_limit = int(upper_limit) + 1
"""

"""
    fig = plt.figure(layout='constrained')

    root = math.ceil(math.sqrt(len(main.value)))

    gs = gridspec.GridSpec(root, root, figure=fig)

    xlabel = second.name
    ylabel = name

    for key, idx in zip(graphs.keys(), itertools.product(range(0, root), range(0, root))):

        x = []
        y = []
        for exp in graphs[key]:
            x.append(exp["Parameters"][second.name])
            y.append(exp["Averages"][name]["avg"])

        zipped = sorted(zip(x, y))

        print(zipped)

        for i in range(len(x)):
            x[i], y[i] = zipped[i] 
        

        ax = fig.add_subplot(gs[idx[0], idx[1]])
    
        x = [str(i) for i in x]

        p = ax.bar(x, y , label=x)
        ax.set_title(f"{main.name} {key}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.bar_label(p, label_type='center')

        ax.set(ylim=(0,upper_limit))


    plt.savefig("AVG_" + name + "_per_" + main.name +"_and_"+second.name+".png")
"""
    



def plot_avg(file, name, main, second, plot_type):
   
    file.seek(0)

    json_dict = json.load(file)

    graphs = dict()

    for x in main.value:
        graphs[x] = list()

    upper_limit = 0

    for exp in json_dict["Experiments"]:
        graphs[exp["Parameters"][main.name]].append(exp)
        if exp["Averages"][name]["avg"] > upper_limit:
            upper_limit = exp["Averages"][name]["avg"]
  
    upper_limit = int(upper_limit) + 1


    fig = plt.figure(layout='constrained')

    root = math.ceil(math.sqrt(len(main.value)))

    gs = gridspec.GridSpec(root, root, figure=fig)

    xlabel = second.name
    ylabel = name

    for key, idx in zip(graphs.keys(), itertools.product(range(0, root), range(0, root))):

        x = []
        y = []
        for exp in graphs[key]:
            x.append(exp["Parameters"][second.name])
            y.append(exp["Averages"][name]["avg"])

        zipped = sorted(zip(x, y))

        print(zipped)

        for i in range(len(x)):
            x[i], y[i] = zipped[i] 
        

        ax = fig.add_subplot(gs[idx[0], idx[1]])
    
        x = [str(i) for i in x]

        p = ax.bar(x, y , label=x)
        ax.set_title(f"{main.name} {key}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.bar_label(p, label_type='center')

        ax.set(ylim=(0,upper_limit))


    plt.savefig("AVG_" + name + "_per_" + main.name +"_and_"+second.name+".png")

def collapse_experiment(exp):
    averages = dict()

    for run in exp["Runs"]:
        for out in run["Outputs"]:
            for key in out.keys():
                if isinstance(out[key], dict) == False:
                    if key not in averages.keys():
                        averages[key] = min_max_avg()
                    averages[key].compute_next(out[key])
                elif out[key]["Type"] == 1:
                    mma = min_max_avg(average=out[key]["Value"][0], 
                                      minimum=out[key]["Value"][1], 
                                      maximum=out[key]["Value"][2], 
                                      count=out[key]["Value"][3])
                    if key not in averages.keys():
                        averages[key] = mma
                    else:
                        averages[key].combine_mma(mma)

    for key in averages.keys():
        averages[key] = averages[key].return_dict()
    
    return averages
