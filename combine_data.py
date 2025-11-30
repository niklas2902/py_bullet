import glob
import json

if __name__ == "__main__":
    from tqdm import tqdm
    collision_data_files = glob.glob("logs/*.json")
    collision_data_files = list(filter(lambda f: "collision_points" not in f, collision_data_files))
    output = []
    for file in tqdm(collision_data_files, desc="Processing JSON files"):

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



    collision_points_files = glob.glob("logs/*.json")
    collision_points_files = list(filter(lambda f: "collision_points" in f, collision_points_files))
    output = []
    for file in tqdm(collision_points_files, desc="Processing JSON files"):

        with open(file) as f:
            data = []
            try:
                data = json.load(f)
            except Exception as e:
                print(e)
            for data_entry in data:
                data_entry["filename"] = file
                output.append(data_entry)


    with open("final_output_contact_points.json", "w") as f:
        json.dump(output, f)
