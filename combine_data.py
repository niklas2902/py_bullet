import glob
import json

if __name__ == "__main__":
    from tqdm import tqdm
    json_files = glob.glob("logs/*.json")
    output = []
    for file in tqdm(json_files, desc="Processing JSON files"):
        with open(file) as f:
            data = []
            try:
                data = json.load(f)
            except Exception as e:
                print(e)
            for data_entry in data:
                data_entry["filename"] = file
                output.append(data_entry)


    with open("final_output.json", "w") as f:
        json.dump(output, f)
