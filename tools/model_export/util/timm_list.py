from timm import list_models

with open("timm_list.txt", "w") as f:
    models=list_models(pretrained=True)
    for model in models:
        f.write(f"{model}\n")
    
