import pm4py


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
# A trace is a sequence of events (each row in the log dataframe is an event in a trace) that have the same 'case:case' column value
def random_swap_events(trace):
    #this is one type of perturbation
    num_rows = len(trace)
    idx1, idx2 = random.sample(range(num_rows), 2)
    trace.iloc[idx1], trace.iloc[idx2] = trace.iloc[idx2], trace.iloc[idx1]
    return trace

def skip_mandatory_activity(trace):
    #this is another type of perturbation
    trace = copy.deepcopy(trace)
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

ignored_columns = ["time", "@@index", "time:timestamp"]
traces = log.groupby('case:case')
trace_ids = traces.groups.keys()
valid_traces = [traces.get_group(trace_id) for trace_id in trace_ids]
invalid_traces = []
for trace in valid_traces:
    candidate = random_swap_events(copy.deepcopy(trace))
    if candidate not in invalid_traces and candidate not in valid_traces:
        invalid_traces.append(candidate)



# get list of traces
#traces = [trace for trace in traces]
breakpoint()
print(traces)