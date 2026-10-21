import pm4py
from copy import deepcopy
import random
from langchain_openai import ChatOpenAI


# A trace is a sequence of events (each row in the log dataframe is an event in a trace) that have the same 'case:case' column value
def random_swap_events(trace, verbose=False):
    #this is one type of perturbation
    perturbed_trace = deepcopy(trace)
    num_rows = len(perturbed_trace)
    idx1, idx2 = random.sample(range(num_rows), 2)
    if verbose: print(f"Swapped {idx1} with {idx2}")
    perturbed_trace.iloc[idx1], perturbed_trace.iloc[idx2] = perturbed_trace.iloc[idx2], perturbed_trace.iloc[idx1]
    return perturbed_trace

def skip_mandatory_activity(trace):
    #this is another type of perturbation
    trace = deepcopy(trace)
    num_rows = len(trace)
    idx = random.sample(range(num_rows), 1)[0]
    trace.drop(idx, inplace=True)
    return trace

def insert_impossible_activity(trace):
    #this is another type of perturbation
    # use an LLM to generate a new activity that is impossible to insert but still sounds like a valid activity
    raise NotImplementedError("Not implemented")

def duplicate_activity(trace):
    #this is another type of perturbation
    #take a random activity from the trace and insert a duplication in a random position of the trace
    raise NotImplementedError("Not implemented")

def add_invalid_loop(trace):
    raise NotImplementedError("Not implemented")


def df_to_key_value_rows(df, max_rows=50):
    df_short = df.head(max_rows).reset_index(drop=True)
    lines = []

    for i, row in df_short.iterrows():
        parts = [f"{col}={row[col]}" for col in df_short.columns]
        lines.append(f"Event number {i}: " + "; ".join(parts))

    return "\n".join(lines)




# sample 100 valid traces
# sample 100 invalid traces
# load LLM with the system prompt and drop
def eval_model(valid_traces, invalid_traces):
    llm = ChatOpenAI(
    model="gpt-5-nano", api_key="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")
    with open("/home/marco/Escritorio/austral/all-things-agents/pm-samples/runs/run_hiring_log_high_20251116_210442/textual_model.txt", "r") as f:
        business_model_description = f.read()
    system_prompt = f"""You are an expert business process analyst. You \
  will be given a business process description and a trace (sequence of \
  activities).\

  Business Process Description:\
  {business_model_description}\
\
  Your task: Determine if the given trace is VALID (conforms to the observed\
   business process) or INVALID (contains anomalies).\
\
  IMPORTANT: The business model describes processes as they are observed, \
  which may include inefficiencies or non-ideal behaviors that still occur \
  in practice.\
\
  Respond with ONLY one word: "VALID" or "INVALID"."""
  
    #breakpoint()
    n_valid_detected, n_invalid_detected = 0, 0
    false_negatives, false_positives = 0, 0
    ilegal_outputs = []
    

    for valid_trace in valid_traces:
        #breakpoint()
        messages = [("system",system_prompt,),("human", f"{df_to_key_value_rows(valid_trace)}"),]
        ai_msg = llm.invoke(messages)
        response = ai_msg.content; print(response)
        if response=="VALID": n_valid_detected+=1
        elif response=="INVALID": false_negatives+=1
        else: ilegal_outputs.append(response)
        print(f"should be VALID, labeled as {str(response)}")
        print(f"False negatives: {str(false_negatives)}")
    for invalid_trace in invalid_traces:
        #breakpoint()
        messages = [("system",system_prompt,),("human", f"{df_to_key_value_rows(invalid_trace)}"),]
        ai_msg = llm.invoke(messages)
        response = ai_msg.content; print(response)
        if response=="INVALID": n_invalid_detected+=1
        elif response=="VALID": false_positives+=1
        else: ilegal_outputs.append(response)
        print(f"should be INVALID, labeled as {str(response)}")
        print(f"False positives: {str(false_positives)}")

    TP = n_valid_detected
    TN = n_invalid_detected
    FN = len(valid_traces)-TP
    FP = len(invalid_traces)-TN
    #assert(FP==false_positives); assert(FP==false_positives)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    print(f"Presicion: {precision}\nRecall: {recall}\
        \nF1: {f1}\n False Positives: {false_positives}\
        \n False Negatives: {false_negatives}\
        TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    #breakpoint()



#generalization

class Benchmark:
    def __init__(self):
        problems = ["hiring", "hospital", "lending", "renting"]


class TSADDatasetGenerator:
    #after "hardcoding" the full evaluation pipeline we will generalize using classes like this one
    def __init__(self, log_path, discard_columns=[], load_cached=True):
        self.log = pm4py.read_xes(log_path).drop(discard_columns,axis=1)
        self.log_path = log_path
        raise NotImplementedError
    def random_sample_valid_traces(self):
        raise NotImplementedError
    def random_sample_invalid_traces(self):
        raise NotImplementedError
    def generate_invalid_traces(self, sample_size="full"):
        raise NotImplementedError

class TSADEvaluator:
    # LLM-based TSAD evaluation
    def __init__(self, llm, business_model):
        self.business_model = business_model
        self.llm = llm
        # the following prompt is a guideline, should be correctly engineered
        self.system_prompt = "You will be given a business model that describes the \
            functioning of its OBSERVED processes (they might not be ideal). \
                Given that business model and a trace, decide if the trace can happen according to the business model...\
                \
                KEEP IN MIND THAT THE BUSINESS MODEL ALSO DESCRIBES TRACES THAT HAPPEN EVEN IF THEY WOULD NOT BE DESIRED IN AN IDEAL ONE"
        #self.dataset = TSAD

def main():
    # load log file from benchmark/problem_name/problem_name_log_high.xes
    problem_logs = [
        "benchmark/hiring/hiring_log_high.xes",
        "benchmark/hospital/hospital_log_high.xes",
        "benchmark/lending/lending_log_high.xes",
        "benchmark/renting/renting_log_high.xes"
    ]

    log = pm4py.read_xes(problem_logs[0])
    #get list of traces

    """
    (Pdb) log.columns
    Index(['activity', 'resource', 'time', 'concept:name', 'time:timestamp',
        '@@index', 'case:concept:name', 'case:german speaking',
        'case:yearsOfEducation', 'case:age', 'case:gender', 'case:citizen',
        'case:protected', 'case:@@case_index', 'case:case', 'case:religious'],
        dtype='object')

    """

    ignored_columns = ["time", "@@index", "time:timestamp"]
    log = log.drop(ignored_columns, axis=1)
    traces = log.groupby('case:case')
    trace_ids = traces.groups.keys()
    valid_traces = [traces.get_group(trace_id) for trace_id in trace_ids]

    valid_traces_300 = random.sample(valid_traces, 300)
    invalid_traces = []



    for trace in valid_traces_300:
        candidate = random_swap_events(deepcopy(trace))
        if all(not candidate.equals(t) for t in invalid_traces) and \
            all(not candidate.equals(t) for t in valid_traces_300):
            #print(f"Invalid Candidate generated: \n{candidate}")
            invalid_traces.append(candidate)
        else: print("Candidate was valid")

    print(len(invalid_traces))

    eval_model(random.sample(valid_traces_300,20),random.sample(invalid_traces,20))

if __name__ == "__main__":
    exit(main())
