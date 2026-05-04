from Conllu_UD_Parser import Conllu_UD_Parser
 
"""
This code is used to run the Conllu_UD_Parser. In the file_mappings you can configure your starting data.
"""
file_mapping = {
    "Romanian_RRT": [
        "ro_rrt-ud-train.conllu",
        "ro_rrt-ud-dev.conllu"
    ],
    "Romanian_Nonstandard": [
        "ro_nonstandard-ud-train.conllu",
        "ro_nonstandard-ud-dev.conllu"
    ]
}
 
parser = Conllu_UD_Parser(file_mapping=file_mapping)
output_paths = parser.process()
print(f"Processing complete. Data saved to: {output_paths}")