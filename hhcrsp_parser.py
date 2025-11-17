
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import re
import pathlib
import matplotlib.pyplot as plt

@dataclass
class HHCRSPInstance:
    name: str
    nb_customers: int   # numero pazienti
    nb_synch: int       # numero pazienti con sincronizzazioni o preferenze
    nb_nurses: int      # numero caregiver
    nb_services: int    # numero di skills
    nb_pref: int        # Preferenze se ci sono (variabile binaria )

    skills: List[List[int]]                 # nurses x skills: entry (i,j) = 1 if nurse i has skill j

    demands: List[List[int]]                # patients x skills: entry (i,j) = 1 if patient i requires skill j
    duration: List[List[int]]               # patients x skills: entry (i,j) is the time required for patient i requiring skill j
    pref: List[List[float]]                 # patients x nurses: entry (i,j) preference of patient i for nurse j (between -10 and 10)

    d_ini_to_nurses: List[List[float]]      # patients x nurses (distance/time from nurse k start location to patient i)
                                            # entrambi dimensione: nb_customers x nb_nurses
    d_fi_to_nurses: List[List[float]]       # patients x nurses (distance/time from patient i to nurse k end location)

    d_customers: List[List[float]]          # patients x patients (pairwise distances/times)

    gap_min_max: List[Tuple[int,int]]       # patients x 2 (min first column,max second column gaps between paired jobs; 0 0 if none)

    tw_nurses: List[Tuple[int,int]]         # nurses x 2 (start,end) minutes, time windows
    tw_customers: List[Tuple[int,int]]      # patients x 2 (start,end) minutes, time windows

    cxy_nurses: List[Tuple[float,float]]    # nurses x 2 (coords; optional). posizione geografica (x, y) di ciascun infermiere, cioè il punto da cui parte e dove ritorna.
    # una nurse impiega lo stesso tempo per andare da sé al paziente e per tornare dal paziente al punto di partenza
    cxy_customer: List[Tuple[float,float]]  # patients x 2 (coords; optional)

    meta: Dict[str, Any]                    # campi extra

def _parse_matrix(lines: List[str]) -> List[List[float]]:
    
    rows: List[List[float]] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.endswith('];'):
            ln = ln[:-2].strip()
            if ln:
                rows.append([float(x) for x in re.split(r'\s+', ln)])
            break
        else:
            rows.append([float(x) for x in re.split(r'\s+', ln)])
    return rows

def _parse_int_pairs(lines: List[str]) -> List[Tuple[int,int]]:
    rows = _parse_matrix(lines)
    return [(int(r[0]), int(r[1])) for r in rows]

def _parse_float_pairs(lines: List[str]) -> List[Tuple[float,float]]:
    rows = _parse_matrix(lines)
    return [(float(r[0]), float(r[1])) for r in rows]

def load_instance(path: str | pathlib.Path) -> HHCRSPInstance:
    
    path = pathlib.Path(path)
    text = path.read_text(encoding='utf-8', errors='ignore')
    # normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Split into tokens marking each section header
    # We'll iterate line-by-line and when a header is matched, we parse the block.
    lines = iter(text.splitlines())
    def next_nonempty():
        for l in lines:
            if l.strip():
                return l
        return None

    header_values: Dict[str, Any] = {}
    blocks: Dict[str, List[List[float]]] = {}
    pairs_blocks: Dict[str, List[Tuple[int,int]]] = {}
    float_pairs_blocks: Dict[str, List[Tuple[float,float]]] = {}
    meta: Dict[str, Any] = {}

    # Precompile regexes
    kv_re = re.compile(r'^(NbCustomers|NbSynch|NbNurses|NbServices|NbPref)\s*=\s*(\d+)\s*$')
    block_re = re.compile(r'^(Skills|Demands|Duration|Pref|d_ini_to_nurses|d_fi_to_nurses|d_customers|gap_min_max|TW_nurses|TW_customers|cxy_nurses|cxy_customer|PercentageNursesPerSkill|PercentagePatientsPerSkill)\s*=\s*$')

    # Convert the iterator into a list so we can manually advance when needed
    line_list = text.splitlines()
    i = 0
    n = len(line_list)

    def collect_block(start_index: int) -> Tuple[List[str], int]:
        """Collect lines starting at start_index (which points to the first data line) up to and including a line with '];'."""
        collected = []
        j = start_index
        while j < n:
            ln = line_list[j].strip()
            collected.append(ln)
            if ln.endswith('];'):
                return collected, j + 1
            j += 1
        # If we reach here, block was not properly closed; still return what we have
        return collected, j

    while i < n:
        raw = line_list[i].strip()
        i += 1
        if not raw:
            continue

        m_kv = kv_re.match(raw)
        if m_kv:
            header_values[m_kv.group(1)] = int(m_kv.group(2))
            continue

        m_blk = block_re.match(raw)
        if m_blk:
            key = m_blk.group(1)
            # next line should be data; collect until '];'
            block_lines, i = collect_block(i)
            if key in ('gap_min_max',):
                pairs_blocks[key] = _parse_int_pairs(block_lines)
            elif key in ('TW_nurses','TW_customers','cxy_nurses','cxy_customer'):
                # They are all 2-column pairs, but tw_* are int pairs, cxy_* are float pairs
                if key.startswith('TW_'):
                    pairs_blocks[key] = _parse_int_pairs(block_lines)
                elif key.startswith('cxy_'):
                    float_pairs_blocks[key] = _parse_float_pairs(block_lines)
            elif key in ('PercentageNursesPerSkill','PercentagePatientsPerSkill'):
                rows = _parse_matrix(block_lines)
                # Flatten single-row percentages
                meta[key] = rows[0] if rows else []
            else:
                blocks[key] = _parse_matrix(block_lines)
            continue

    # Sanity conversions and casting shapes
    def to_int_matrix(m): return [[int(x) for x in row] for row in m]
    def to_int_pairs(p): return [(int(a), int(b)) for a,b in p]

    nb_customers = int(header_values.get('NbCustomers', 0))
    nb_synch = int(header_values.get('NbSynch', 0))
    nb_nurses = int(header_values.get('NbNurses', 0))
    nb_services = int(header_values.get('NbServices', 0))
    nb_pref = int(header_values.get('NbPref', 0))

    skills = to_int_matrix(blocks.get('Skills', []))
    demands = to_int_matrix(blocks.get('Demands', []))
    duration = to_int_matrix(blocks.get('Duration', []))
    pref = blocks.get('Pref', [])
    d_ini_to_nurses = blocks.get('d_ini_to_nurses', [])
    d_fi_to_nurses = blocks.get('d_fi_to_nurses', [])
    d_customers = blocks.get('d_customers', [])
    gap_min_max = to_int_pairs(pairs_blocks.get('gap_min_max', []))
    tw_nurses = to_int_pairs(pairs_blocks.get('TW_nurses', []))
    tw_customers = to_int_pairs(pairs_blocks.get('TW_customers', []))
    cxy_nurses = float_pairs_blocks.get('cxy_nurses', [])
    cxy_customer = float_pairs_blocks.get('cxy_customer', [])

    # Basic validation
    if nb_customers and len(d_customers) and len(d_customers) != nb_customers:
        raise ValueError(f"Distance matrix size {len(d_customers)} != NbCustomers {nb_customers}")
    if nb_nurses and len(skills) and len(skills) != nb_nurses:
        raise ValueError(f"Skills rows {len(skills)} != NbNurses {nb_nurses}")
    if nb_services and len(skills) and len(skills[0]) != nb_services:
        raise ValueError(f"Skills cols {len(skills[0])} != NbServices {nb_services}")
    if nb_customers and len(demands) and len(demands) != nb_customers:
        raise ValueError(f"Demands rows {len(demands)} != NbCustomers {nb_customers}")

    return HHCRSPInstance(
        name=pathlib.Path(path).name,
        nb_customers=nb_customers,
        nb_synch=nb_synch,
        nb_nurses=nb_nurses,
        nb_services=nb_services,
        nb_pref=nb_pref,
        skills=skills,
        demands=demands,
        duration=duration,
        pref=pref,
        d_ini_to_nurses=d_ini_to_nurses,
        d_fi_to_nurses=d_fi_to_nurses,
        d_customers=d_customers,
        gap_min_max=gap_min_max,
        tw_nurses=tw_nurses,
        tw_customers=tw_customers,
        cxy_nurses=cxy_nurses,
        cxy_customer=cxy_customer,
        meta=meta,
    )





