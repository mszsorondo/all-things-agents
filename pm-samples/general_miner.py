import pm4py



#metadata
h_billing_description = open('benchmark/Hospital Billing - Event Log_1_all/readme.txt', 'r').read()

#logs
h_billing_log = pm4py.read_xes("benchmark/Hospital Billing - Event Log_1_all/Hospital Billing - Event Log.xes")

breakpoint()
#miners
h_billing_bpmn = pm4py.discover_bpmn_inductive(h_billing_log)
h_billing_heu_net = pm4py.discover_heuristics_net(h_billing_log)
h_billing_llm_dfg = pm4py.llm.abstract_dfg(h_billing_log)

conceptual_model = pm4py.llm.openai_query(f"The following is a description of a business process along with its corresponding DFG model obtained from logs by doing Process Mining on them. Generate a description of the process based exclusively on the provided data. \nDescription: {h_billing_description} \nDFG:{h_billing_llm_dfg}", "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")


#viewers
pm4py.view_bpmn(h_billing_bpmn)
pm4py.view_heuristics_net(h_billing_heu_net)
