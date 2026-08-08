"""BM25 en Python puro. Sin dependencias, sin descargas, deterministico."""
import math, re, unicodedata
from collections import Counter

K1, B = 1.5, 0.75

VACIAS = {"de","la","el","los","las","y","o","a","en","que","con","por","para","del",
          "se","un","una","al","es","su","sus","lo","como","mas","si","no","ni",
          "the","of","and","in","to","for","is","on","with","this","that","are","be"}

def tokenizar(t: str):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return [w for w in re.findall(r"[a-z0-9]{3,}", t) if w not in VACIAS]

class BM25:
    def __init__(self):
        self.docs = []
        self.tf = []
        self.df = Counter()
        self.long = []
        self.avg = 0.0

    def indexar(self, docs):
        self.docs = docs
        self.tf, self.long, self.df = [], [], Counter()
        for d in docs:
            c = Counter(tokenizar(d["texto"]))
            self.tf.append(c); self.long.append(sum(c.values()))
            for w in c: self.df[w] += 1
        self.avg = (sum(self.long) / len(self.long)) if self.long else 0.0

    def buscar(self, consulta, k=3, filtro=None):
        q = tokenizar(consulta)
        N = len(self.docs)
        if not q or N == 0: return []
        idf = {w: math.log(1 + (N - self.df.get(w, 0) + 0.5) / (self.df.get(w, 0) + 0.5))
               for w in set(q)}
        out = []
        for i, d in enumerate(self.docs):
            if filtro and not filtro(d): continue
            s, L = 0.0, self.long[i]
            for w in q:
                f = self.tf[i].get(w, 0)
                if not f: continue
                s += idf[w] * (f * (K1 + 1)) / (f + K1 * (1 - B + B * L / (self.avg or 1)))
            if s > 0: out.append((s, d))
        out.sort(key=lambda x: -x[0])
        return [{"score": round(s, 4), **d} for s, d in out[:k]]
