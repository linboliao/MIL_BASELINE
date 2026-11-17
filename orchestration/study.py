"""Immutable experiment identities and NAS-backed study definitions."""
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class Cell:
    pfm: str
    domain: str
    config: Path
    splits: Path

    @property
    def key(self):
        return self.pfm + '/' + self.domain


@dataclass(frozen=True)
class Study:
    name: str
    source: Path
    output_parent: Path
    pfms: tuple
    domains: tuple

    @classmethod
    def load(cls, path):
        spec = json.loads(Path(path).read_text())
        source = Path(spec['source']).resolve(strict=True)
        parent = Path(spec['output_parent'])
        provenance = Path('/NAS145/linboliao/Experiments/ProstateDiagnosis/provenance')
        if provenance not in parent.resolve().parents:
            raise ValueError('output_parent must be below project provenance')
        for field in ('pfms', 'domains'):
            values = spec[field]
            if not values or len(set(values)) != len(values):
                raise ValueError('Empty or duplicate ' + field)
            if any(not isinstance(v, str) or not v or '/' in v or v in ('.', '..') for v in values):
                raise ValueError('Invalid ' + field)
        return cls(spec['name'], source, parent, tuple(spec['pfms']), tuple(spec['domains']))

    def cells(self, pfms=None, domains=None):
        pfms = self.pfms if pfms is None else tuple(pfms)
        domains = self.domains if domains is None else tuple(domains)
        if not pfms or not domains or len(set(pfms)) != len(pfms) or len(set(domains)) != len(domains):
            raise ValueError('Empty or duplicate selection')
        if set(pfms)-set(self.pfms) or set(domains)-set(self.domains):
            raise ValueError('Unknown PFM or domain')
        return tuple(Cell(p, d, self.source/p/d/'configs/model.yaml', self.source/p/d/'splits')
                     for p in pfms for d in domains)
