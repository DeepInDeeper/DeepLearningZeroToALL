import json

def load_data(submit_file, reference_file):
  # load submit result and reference result

    with open(submit_file, 'r') as file1:
        submit_data = json.load(file1)
    with open(reference_file, 'r') as file1:
        ref_data = json.load(file1)
    if len(submit_data) != len(ref_data):
        print('Inconsistent number of images between submission and reference data \n')
    submit_dict = {}
    ref_dict = {}
    for item in submit_data:
        submit_dict[item['image_id']] = item['label_id']
    for item in ref_data:
        ref_dict[item['image_id']] = item['label_id']
    return submit_dict, ref_dict


submit_file = "result.json"
ref_file = "submit.json"
dict_data,ref_data = load_data(submit_file,ref_file)
print(len(dict_data),len(ref_data))
#print (dict_data)


