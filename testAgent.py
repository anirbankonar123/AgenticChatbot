from multi_doc_agent import MultiDocAgent

multiDocAgent=MultiDocAgent()
agent=multiDocAgent.get_agent("data")

response = agent.query("What is the mass of moon in kg ?")

#response = agent.query("What is the actual value of atmospheric pressure on the day side of the Moon ?")

#response = agent.query("What are the experiments onboard Chandrayaan 3 ?")

print(response)
# print("#################")
# ctr=0
with open("output.txt","a") as f:
    for node in response.source_nodes:
        f.writelines(str(node.metadata))
f.close()

