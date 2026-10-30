def load_txt_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data 