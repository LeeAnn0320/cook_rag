from pathlib import Path

data_path="D:\Data_Study\LLM_study\cook_rag\data"
data_path_obj=Path(data_path)
for md_file in data_path_obj.rglob("*.md"):
    data_root=Path(data_path).resolve()
    file_path=Path(str(md_file))
    print(file_path)
    path_parts=file_path.parts
    print(path_parts)
    print(file_path.stem)
    print(data_root)
    relative_path=Path(md_file).resolve().relative_to(data_root).as_posix()
    print(relative_path)
    break