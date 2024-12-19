import json
import argparse
import sys
import numpy as np
import os

def add_layer(layer_name, layer_data, header_data, size):
    """
    Adds layer weights and biases to header_data
    """
    line = ""
    if layer_name in ["rec.weight_ih_l0", "rec.weight_hh_l0", "lin.weight"]:
        var_declaration = "  Model." + layer_name.replace(".", "_") + " = "
    else:
        var_declaration = "  Model." + layer_name.replace(".", "_") + " = "
    line += var_declaration

    line += "{"
    c = 0

    if len(np.asarray(layer_data).shape) > 1:
        for i in layer_data:
            c += 1
            if c == 1:
                line += "{"
            else:
                line += " " * len(var_declaration) + " { "
            c2 = 0
            i_rzSwap = np.concatenate((i[size:size*2], i[0:size], i[size*2:]))
            for j in i_rzSwap:
                c2 += 1
                if c2 == len(i):
                    line += str(j)
                else:
                    line += str(j) + ", "
            if c == len(layer_data):
                line += "}}; "
            else:
                line += "}, "
            header_data.append(line)
            line = ""
    else:
        for i in layer_data:
            c += 1
            line += str(i)
            if c == len(layer_data):
                line += "}; "
            else:
                line += ", "
        header_data.append(line)

    header_data.append("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_model", type=str, default="", help="Path to json file, including filename")
    args = parser.parse_args()

    # Read target JSON file
    with open(args.json_model) as f:
        data = json.load(f)

    header_data = []
    header_data.append("//========================================================================")
    header_data.append("//" + args.json_model.split(".json")[0])
    header_data.append("/*")
    for item in data['model_data'].keys():
        header_data.append(item + " : " + str(data['model_data'][item]))
    header_data.append("*/\n")
    size = int(data['model_data']['hidden_size'])

    for layer_name in data['state_dict'].keys():
        if layer_name.startswith("rec.bias_"):
            continue
        if layer_name in ["rec.weight_ih_l0", "rec.weight_hh_l0"]:
            add_layer(layer_name, np.array(data['state_dict'][layer_name]).T, header_data, size)
        else:
            add_layer(layer_name, data['state_dict'][layer_name], header_data, size)

    gru_bias = np.array([np.array(data['state_dict']['rec.bias_ih_l0']), np.array(data['state_dict']['rec.bias_hh_l0'])])
    add_layer('rec_bias', gru_bias, header_data, size)

    # Create the /headers directory if it doesn't exist
    output_dir = "./headers"
    os.makedirs(output_dir, exist_ok=True)

    # Extract the directory name (e.g., 'DS1') for the header file name
    dir_name = os.path.basename(os.path.dirname(args.json_model))
    new_filename = os.path.join(output_dir, f"{dir_name}.h")

    # Write data to .h file in /headers
    with open(new_filename, 'w') as file:
        for item in header_data:
            file.write(item + "\n")

    print("Finished generating header file: " + new_filename)
