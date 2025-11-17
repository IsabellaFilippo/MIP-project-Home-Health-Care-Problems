
# python scalability.py folder/with/instances

import sys, glob, os, time, csv
from hhcrsp_parser import load_instance
from test import build_model

def main():
    if len(sys.argv) < 2:
        print("Us0: python scalability.py <instances_folder>")
        return
    folder = sys.argv[1]
    tlim = 300

    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not paths:
        print("No *.txt instances found.")
        return

    out_csv = os.path.join(folder, "scalability_results.csv")
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["instance","n_customers","n_nurses","build_sec","solve_sec","status","obj","gap"])
        for p in paths:
            t0 = time.time()
            inst = load_instance(p)
            m = build_model(inst, weights=dict(travel=0.5, pref=0.5, tard_total=0, tard_max=0.0), time_limit=tlim)
            t1 = time.time()
            m.optimize()
            t2 = time.time()
            status = m.Status
            obj = m.ObjVal if status in (2, 9) else ""
            gap = m.MIPGap if hasattr(m, "MIPGap") and status in (2, 9) else ""
            wr.writerow([os.path.basename(p), inst.nb_customers, inst.nb_nurses, f"{t1-t0:.2f}", f"{t2-t1:.2f}", status, obj, gap])
            print(f"Done {os.path.basename(p)}: status={status}, obj={obj}, gap={gap}")

    print(f"Risultati salvati in {out_csv} nella cartella corrente.")

if __name__ == "__main__":
    main()
