# MILP per Home Health Care Routing & Scheduling Problem (HHCRSP)
from __future__ import annotations
import sys
from hhcrsp_parser import load_instance
import gurobipy as gp
from gurobipy import GRB # costanti di Gurobi
from typing import Dict, Any, Tuple
from hhcrsp_parser import HHCRSPInstance

def build_model(inst: HHCRSPInstance, weights, time_limit = 300, params: Dict[str,Any] | None = None) -> gp.Model:

    weights_diff = dict(travel=0.5, pref=0.5, tard_total=0, tard_max=0)
    if weights:
        weights_diff.update(weights)


    m = gp.Model("HHCRSP")

    # Indici di base
    I = range(inst.nb_customers)   # pazienti
    K = range(inst.nb_nurses)      # caregivers
    S = range(inst.nb_services)    # servizi

    # Insieme dei Job
    J = [(i, s) for i in I for s in S if inst.demands[i][s] == 1]

    # Caregiver abilitati al job, insieme K_i
    # per ogni job j=(i,s) voglio la lista dei caregiver che possono farlo
    K_of_J = {}   # dizionario: key = (i,s) , value = lista di k
    for (i, s) in J: # paziente, servizio richiesto
        K_of_J[(i, s)] = [k for k in K if inst.skills[k][s] == 1]

    # Per ogni caregiver quali job, per i vincoli 6 e 7
    J_of_K = {k: [] for k in K}
    for (i, s) in J:
        for k in K_of_J[(i, s)]:
            J_of_K[k].append((i, s))

    # L serve per definizione di x
    # L = { (i,j) : esiste almeno un caregiver che può fare i e anche j }
    L = []
    for j1 in J:
        for j2 in J:
            if j1 == j2:
                continue
            Ki = set(K_of_J[j1])
            Kj = set(K_of_J[j2])
            if Ki.intersection(Kj):
                L.append((j1, j2))

    # Insieme R per la definizione di z
    R = []          # lista delle coppie di job con relazione
    gmin = {}       # mappa: (job1, job2) -> gmin
    gmax = {}       # mappa: (job1, job2) -> gmax

    for i in range(inst.nb_customers):
        # tutti i servizi richiesti dal paziente i
        services_i = [s for s in range(inst.nb_services) if inst.demands[i][s] == 1]

        # se chiede meno di 2 servizi, non c'è relazione
        if len(services_i) < 2:
            continue

        # prendo il gap per questo paziente
        gap_i_min, gap_i_max = inst.gap_min_max[i]   # es. (0,0) oppure (30,60)

        # ora faccio tutte le coppie di servizi di QUESTO paziente
        for idx1 in range(len(services_i)):
            for idx2 in range(idx1 + 1, len(services_i)):
                s1 = services_i[idx1]
                s2 = services_i[idx2]

                job1 = (i, s1)
                job2 = (i, s2)

                # aggiungo la coppia a R
                R.append((job1, job2))
                gmin[(job1, job2)] = gap_i_min
                gmax[(job1, job2)] = gap_i_max

    # calcolo B_j per ogni job j=(i,s) per 14
    B = {}
    for (i, s) in J:
        candidates = []
        for k in K_of_J[(i, s)]:
            beta_k = inst.tw_nurses[k][1]                 # fine turno caregiver k
            dur_is = inst.duration[i][s]                  # durata del job (i,s)
            back_time = inst.d_fi_to_nurses[i][k]         # tempo per tornare dal paziente i al caregiver k
            candidates.append(beta_k - dur_is - back_time)
        # se per qualche motivo non ci sono caregiver, prendo un B grande
        B[(i, s)] = max(candidates) if candidates else 1e5 #100000

    # ----------------------
    # Variabili binarie
    # ----------------------

    # x[i,j] = 1 se j viene subito dopo i in un route (di qualche caregiver)
    x = m.addVars(L, vtype=GRB.BINARY, name="x")

    # z[i,y] = 1 se il job i viene prima di y
    z = m.addVars(R, vtype=GRB.BINARY, name="z")

    # y[(i,s), k] = 1 se il job (i,s) è eseguito da k
    # NB se il caregiver k non ha la skill per s, non creo la variabile
    y = m.addVars(
        (((i, s), k) for (i, s) in J for k in K_of_J[(i, s)]),
        vtype=GRB.BINARY,
        name="y")

        # o[(i,s), k] = 1 se (i,s) è il primo job di k
    o = m.addVars(
        (((i, s), k) for (i, s) in J for k in K_of_J[(i, s)]),
        vtype=GRB.BINARY,
        name="o"
    )

    # e[(i,s), k] = 1 se (i,s) è l'ultimo job di k
    e = m.addVars(
        (((i, s), k) for (i, s) in J for k in K_of_J[(i, s)]),
        vtype=GRB.BINARY,
        name="e"
    )

    # h[k] = 1 se il caregiver k non fa nessun job
    h = m.addVars(K, vtype=GRB.BINARY, name="h")

    # ----------------------
    # variabili continue
    # ----------------------

    # t[(i,s)] = start time del job (i,s)
    t = m.addVars(J, vtype=GRB.CONTINUOUS, lb=0.0, name="t")

    # w[(i,s)] = ritardo del job (i,s)
    w = m.addVars(J, vtype=GRB.CONTINUOUS, lb=0.0, name="w")

    # massimo ritardo
    W_max = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="W_max")


    # Vincoli

    # 5
    for job in J:
        m.addConstr(W_max >= w[job], name=f"max_tard[{job}]")

    # 6-7

    for k in K:
        # (6) primo job oppure inattivo
        m.addConstr(
            gp.quicksum(o[j, k] for j in J_of_K[k]) + h[k] == 1,
            name=f"first_or_inactive[{k}]"
        )
        # (7) ultimo job oppure inattivo
        m.addConstr(
            gp.quicksum(e[j, k] for j in J_of_K[k]) + h[k] == 1,
            name=f"last_or_inactive[{k}]"
        )

    # 8-9 (simmetria)
    for j in J:  # j è un job, tipo (i,s)
        # tutti i job che possono venire prima di j: cioè tutti gli i con (i, j) in L
        predecessors = [i for (i, jj) in L if jj == j]

        m.addConstr(
            gp.quicksum(x[i, j] for i in predecessors) +
            gp.quicksum(o[j, k] for k in K_of_J[j]) == 1,
            name=f"in_degree[{j}]"
        )

    # (9) ogni job ha esattamente un'uscita (va a un altro job oppure è ultimo)
    for i in J:
        # tutti i job che possono venire dopo i: cioè tutti i j con (i, j) in L
        successors = [j for (ii, j) in L if ii == i]

        m.addConstr(
            gp.quicksum(x[i, j] for j in successors) +
            gp.quicksum(e[i, k] for k in K_of_J[i]) == 1,
            name=f"out_degree[{i}]"
        )

    # 10
    for i in J:
        m.addConstr(
            gp.quicksum(y[i, k] for k in K_of_J[i]) == 1,
            name=f"assign_job[{i}]"
        )

    # 11-12-13

    # (11) se un job è primo o ultimo per k, allora deve essere assegnato a k
    for k in K:
        for j in J_of_K[k]:
            m.addConstr(
                o[j, k] + e[j, k] <= 2 * y[j, k],
                name=f"first_last_implies_assign[{j},{k}]"
            )

    # (12) se k può fare i e j, e c'è un arco tra i e j,
    # allora se i è di k anche j deve essere di k
    for (i_job, j_job) in L:  # tutte le coppie ammissibili
        # caregiver che possono fare ENTRAMBI
        common_ks = set(K_of_J[i_job]).intersection(K_of_J[j_job])
        for k in common_ks:
            m.addConstr(
                y[i_job, k] + x[i_job, j_job] + x[j_job, i_job] <= 1 + y[j_job, k],
                name=f"same_caregiver_on_arc[{i_job},{j_job},{k}]"
            )

    # (13) se k può fare i ma NON può fare j,
    # allora non posso collegare i e j se i è assegnato a k
    for (i_job, j_job) in L:
        Ki = set(K_of_J[i_job])
        Kj = set(K_of_J[j_job])

        # caregiver che possono fare i ma NON j
        only_i = Ki.difference(Kj)

        for k in only_i:
            m.addConstr(
                y[i_job, k] + x[i_job, j_job] + x[j_job, i_job] <= 1,
                name=f"forbid_arc_if_k_cant_do_j[{i_job},{j_job},{k}]"
            )
    

    # 14-15-16 (tempo)
        
    # (14) vincoli di tempo sul sequenziamento
    for (i_job, j_job) in L:   # L è la lista degli archi possibili
        (i, s) = i_job
        (j, s2) = j_job

        travel_ij = inst.d_customers[i][j]        # tempo di viaggio i -> j
        service_i = inst.duration[i][s]           # durata del job i
        bigM = B[i_job]                           # il B_i che abbiamo calcolato prima

        m.addConstr(
            t[i_job] + (service_i + travel_ij) * x[i_job, j_job] <= t[j_job] + bigM * (1 - x[i_job, j_job]),
            name=f"time_link[{i_job}->{j_job}]"
        )

    # (15) chiusura entro fine turno / possibilità di continuare
    for i_job in J:
        (i, s) = i_job
        service_i = inst.duration[i][s]

        # successori di i_job
        successors = [j_job for (ii, j_job) in L if ii == i_job]

        # parte sinistra
        lhs = t[i_job] + service_i
        # se è ultimo di k → aggiungo tempo di rientro
        lhs += gp.quicksum(inst.d_fi_to_nurses[i][k] * e[i_job, k] for k in K_of_J[i_job])
        # se vado a un successore → aggiungo viaggio
        lhs += gp.quicksum(inst.d_customers[i][j_job[0]] * x[i_job, j_job] for j_job in successors)

        # parte destra
        rhs = gp.quicksum(inst.tw_nurses[k][1] * e[i_job, k] for k in K_of_J[i_job]) \
            + gp.quicksum(B[j_job] * x[i_job, j_job] for j_job in successors)

        m.addConstr(lhs <= rhs, name=f"closure_time[{i_job}]")

    # (16) vincolo di inizio: devo poter arrivare in tempo al job i
    for i_job in J:
        (i, s) = i_job

        # predecessori di i_job
        predecessors = [j_job for (j_job, ii) in L if ii == i_job]

        lhs = t[i_job]
        # se è primo di k → tolgo il tempo di andata
        lhs -= gp.quicksum(inst.d_ini_to_nurses[i][k] * o[i_job, k] for k in K_of_J[i_job])
        # se arrivo da j → tolgo tempo di j + viaggio j->i
        lhs -= gp.quicksum(
            (inst.d_customers[j_job[0]][i] + inst.duration[j_job[0]][j_job[1]]) * x[j_job, i_job]
            for j_job in predecessors
        )

        rhs = gp.quicksum(inst.tw_nurses[k][0] * o[i_job, k] for k in K_of_J[i_job]) \
            + gp.quicksum(inst.tw_customers[i][0] * x[j_job, i_job] for j_job in predecessors)

        m.addConstr(lhs >= rhs, name=f"start_time[{i_job}]")


    # 17-18 (precedenza o sincronizzazione)

    for (job1, job2) in R:
        gm = gmin[(job1, job2)]
        gM = gmax[(job1, job2)]

        # job1 = (i, s1)
        # job2 = (i, s2)

        # (17)  (2z - 1) * gmin <= t2 - t1
        m.addConstr(
            (2 * z[job1, job2] - 1) * gm <= t[job2] - t[job1],
            name=f"precedence_min[{job1}->{job2}]"
        )

        # (18)  t2 - t1 <= (2z - 1) * gmax
        m.addConstr(
            t[job2] - t[job1] <= (2 * z[job1, job2] - 1) * gM,
            name=f"precedence_max[{job1}->{job2}]"
        )
    

    # 19-20 (finestre del paziente)
    # (19) se k inizia dopo la chiusura della finestra genera ritardo
    for (i, s) in J:
        b_i = inst.tw_customers[i][1]   # chiusura finestra paziente i
        m.addConstr(
            t[(i, s)] <= b_i + w[(i, s)],
            name=f"tw_close[{i},{s}]"
        )

    # (20) il servizio non può iniziare prima che il paziente sia pronto
    for (i, s) in J:
        a_i = inst.tw_customers[i][0]   # apertura finestra paziente i
        m.addConstr(
            t[(i, s)] >= a_i,
            name=f"tw_open[{i},{s}]"
        )

    # ----------------------
    # Valid inequalities
    # ----------------------

    # durata turno caregiver k
    delta = {k: inst.tw_nurses[k][1] - inst.tw_nurses[k][0] for k in K}   # δ_k

    # l_{k,i} = tempo minimo che k impiega per fare il job i
    # = min(tempo per arrivare al job, tempo per arrivare da un precedente) + durata
    # tempo per arrivare dal depot
    l = {}  # l[(k, job)] = float
    for (i, s) in J:
        for k in K_of_J[(i, s)]:
            # tempo di andata dal caregiver k al paziente i
            to_i = inst.d_ini_to_nurses[i][k]
            # durata del job
            dur = inst.duration[i][s]
            l[(k, (i, s))] = to_i + dur

    Lk = {}
    for k in K:
        start_k, end_k = inst.tw_nurses[k]
        Lk[k] = end_k - start_k   # durata del turno di k


    # (21)
    for k in K:
        m.addConstr(
            gp.quicksum(l[(k, j)] * y[j, k] for j in J_of_K[k]) <=
            gp.quicksum((delta[k] - inst.d_fi_to_nurses[j[0]][k]) * e[j, k] for j in J_of_K[k]),
            name=f"valid_21[{k}]"
        )

    # (22)
    for k in K:
        for i_job in J_of_K[k]:
            (i, s) = i_job

            # predecessori di i_job che k può fare
            preds_k = [
                j_job for j_job in J_of_K[k]
                if (j_job, i_job) in L    # esiste arco j->i
            ]

            lhs = gp.quicksum(l[(k, r_job)] * y[r_job, k] for r_job in J_of_K[k])

            rhs = l[(k, i_job)] * e[i_job, k] \
                + gp.quicksum(
                    ( (delta[k] - inst.d_fi_to_nurses[i][k])      # (δ_k - T^e_{ik})
                    - (inst.d_customers[j_job[0]][i]            # -T_{ji}
                        + inst.duration[i][s]) )                 # -D_i
                    * x[j_job, i_job]
                    for j_job in preds_k
                ) \
                + Lk[k] * (
                    2
                    - gp.quicksum(x[j_job, i_job] for j_job in preds_k)
                    - e[i_job, k]
                )

            m.addConstr(lhs <= rhs, name=f"valid_22[{k},{i_job}]")

    # (26)
    for k in K:
        for i_job in J_of_K[k]:
            (i, s) = i_job
            m.addConstr(
                gp.quicksum(l[(k, j_job)] * y[j_job, k] for j_job in J_of_K[k])
                - inst.duration[i][s] * e[i_job, k]
                <= t[i_job] + Lk[k] * (1 - e[i_job, k]),
                name=f"valid_26[{k},{i_job}]"
            )

    # ----------------------
    # costruzione termini funzione
    # ----------------------

    # T
    # 1) travel sugli archi job->job
    travel_jobs = gp.quicksum(
        inst.d_customers[i_job[0]][j_job[0]] * x[i_job, j_job]
        for (i_job, j_job) in L
    )

    # 2) travel da nurse -> primo job e da ultimo job -> nurse
    travel_depots = gp.quicksum(
        inst.d_ini_to_nurses[i][k] * o[(i, s), k]
        + inst.d_fi_to_nurses[i][k] * e[(i, s), k]
        for (i, s) in J
        for k in K_of_J[(i, s)]
    )

    T_expr = travel_jobs + travel_depots

    # P
    P_expr = gp.quicksum(
        inst.pref[i][k] * y[(i, s), k]
        for (i, s) in J
        for k in K_of_J[(i, s)]
    )

    # W
    W_expr = gp.quicksum(w[job] for job in J)

    # ----------------------
    # funzione obiettivo
    # ----------------------

    m.setObjective(weights_diff['travel']*T_expr + weights_diff['pref']*P_expr + weights_diff['tard_total']*W_expr + weights_diff['tard_max']*W_max, GRB.MINIMIZE)

    # Parametri Gurobi
    m.Params.OutputFlag = 1     # attiva output solver
    if time_limit is not None:
        m.Params.TimeLimit = time_limit # imposta time limit
    
    m.Params.MIPFocus = 1           # cerca prima una soluzione fattibile
    m.Params.Heuristics = 0.2       # di default è 0.05, Gurobi usa euristiche
    m.Params.Cuts = 2               # vincoli creati da Gurobi
    m.Params.PoolGap = 0.0          # solo soluzioni ottimali (0%) o ammetti gap maggiore


    if params:
        for k,v in params.items():
            setattr(m.Params, k, v)

    return m

def main():
    inst_path = "istances/A_10_2_4_6_1_0_0_0.txt"
    inst = load_instance(inst_path)
    print(f"Istanze caricata: {inst.name}")
    print(f"Clienti: {inst.nb_customers}, Infermieri: {inst.nb_nurses}, Servizi: {inst.nb_services}")
    m = build_model(inst, weights=dict(travel=0.5, pref=0.5, tard_total=0, tard_max=0.0), time_limit=600)
    m.optimize()
    status = m.Status
    print(f"Status: {status} ({gp.GRB.Status.getname(status) if hasattr(gp.GRB.Status,'getname') else status})")
    if status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) and m.SolCount > 0:
        print(f"Obj: {m.ObjVal}")
        # Print assignment and a simple route per nurse
        y_vars = {(i,k): m.getVarByName(f'y[{i},{k}]').X for i in range(inst.nb_customers) for k in range(inst.nb_nurses) if m.getVarByName(f'y[{i},{k}]') is not None}
        for k in range(inst.nb_nurses):
            assigned = [i for (i,kk),val in y_vars.items() if kk==k and val>0.5]
            if not assigned:
                continue
            print(f"Infermiere {k}: {len(assigned)} clienti")
    else:
        print("Nessuna soluzione trovata; aumentare il time limit.")

if __name__ == "__main__":
    main()

