import glob

for fp in glob.glob("/scratch/hvp2011/implement/dfr/dfr_group_DRO/outputs/**/train.log", recursive=True):
    if fp.endswith("train.log"):
        with open(fp, "r") as f:
            lines = f.readlines()
        for line_idx in range(len(lines)):

            if lines[line_idx].startswith("{'accuracy_0_0':"):
                lines[line_idx] = eval(lines[line_idx])
                try:
                    lines[line_idx]["mean"] = (lines[line_idx]['worst_accuracy']+lines[line_idx]['avg_accuracy'])/2
                except:
                    lines[line_idx]["mean"] = (lines[line_idx]['worst_accuracy']+lines[line_idx]['mean_accuracy'])/2
                lines[line_idx] = str(lines[line_idx])
        with open(fp.replace(".log", "_processed.log"), "w") as f:
            f.write("\n".join(lines))