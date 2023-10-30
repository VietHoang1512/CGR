import glob

for fp in glob.glob("/scratch/hvp2011/implement/dfr/dfr_group_DRO/outputs/data_dir=/vast/hvp2011/data/waterbird_complete95_forest2water2/test_data_dir=None/data_transform=WaterbirdsForBigCLIPTransform/**/train.log", recursive=True):
    if fp.endswith("train.log"):
        with open(fp, "r") as f:
            lines = f.readlines()
        for line_idx in range(len(lines)):

            if lines[line_idx].startswith("{'accuracy_0_0':"):

                try:
                    lines[line_idx] = eval(lines[line_idx])
                    worst_acc = lines[line_idx]['worst_accuracy']                    
                    avg_acc = lines[line_idx]['avg_accuracy']
                    if(worst_acc>0.9045) and (avg_acc>0.9685):
                        if "test" in lines[line_idx-2]:
                            lines[line_idx] = str(lines[line_idx]) + " test OK"
                        else:
                            lines[line_idx] = str(lines[line_idx]) + " val OK"
                except:
                    pass
                lines[line_idx] = str(lines[line_idx])
        with open(fp.replace(".log", "_processed.log"), "w") as f:
            f.write("\n".join(lines))