import os

def replace_in_file(file_path, old_text, new_text):
    with open(file_path, 'r') as file:
        content = file.read()
    
    content = content.replace(old_text, new_text)
    
    with open(file_path, 'w') as file:
        file.write(content)

def search_and_replace(root_dir, old_text, new_text):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".sh") or file.endswith(".py"):
                file_path = os.path.join(subdir, file)
                replace_in_file(file_path, old_text, new_text)

if __name__ == "__main__":
    root_directory = "/Users/muhammedsaeed/Downloads/From-Nile-s-Bank-to-Digital-Hand"  # Change this to the directory you want to start from
    paths_to_replace = ["PATH TO", "PATH TO"]
    new_path = "PATH TO"
    
    for old_path in paths_to_replace:
        search_and_replace(root_directory, old_path, new_path)
