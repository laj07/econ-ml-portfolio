"""
ESCO skills taxonomy and job-postings loader for the labour market experiment.

ESCO v1.2 (European Skills, Competences, Qualifications and Occupations) is a
multilingual taxonomy of ~14 000 skills and ~3 000 occupations.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class ESCOSkills:
    """
    Loads the ESCO skills taxonomy from the official CSV export.

    Args:
        data_dir: directory containing the ESCO CSV files
                  (skills_en.csv, occupations_en.csv, skills_hierarchy_en.csv)
    """

    def __init__(self, data_dir: str | Path = "data/esco") -> None:
        self.data_dir = Path(data_dir)
        self._skills: pd.DataFrame | None = None

    def load_skills(self) -> pd.DataFrame:
        path = self.data_dir / "skills_en.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"ESCO skills file not found: {path}\n"
                "Run: python scripts/download_esco.py --out data/esco"
            )
        df = pd.read_csv(path)
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        self._skills = df
        log.info("ESCO skills loaded: %d entries", len(df))
        return df

    def skill_labels(self) -> list[str]:
        if self._skills is None:
            self.load_skills()
        return self._skills["preferred_label"].dropna().tolist()

    def skill_descriptions(self) -> list[str]:
        if self._skills is None:
            self.load_skills()
        col = "description" if "description" in self._skills.columns else "preferred_label"
        return self._skills[col].fillna("").tolist()


class JobPostingsDataset:
    """
    Loads a job-postings CSV and pairs it with ESCO labels.

    Expects columns: job_title, description, skills_required (semicolon-sep)

    Args:
        postings_csv: path to job postings file
        esco:         ESCOSkills instance
    """

    def __init__(
        self,
        postings_csv: str | Path,
        esco: ESCOSkills,
    ) -> None:
        self.df   = pd.read_csv(postings_csv)
        self.esco = esco
        log.info("JobPostingsDataset: %d postings", len(self.df))

    def posting_texts(self) -> list[str]:
        """Return full text to embed: title + description."""
        return (self.df["job_title"].fillna("") + ". " +
                self.df["description"].fillna("")).tolist()

    def required_skills(self) -> list[list[str]]:
        """Return list-of-lists of required skill strings per posting."""
        col = "skills_required"
        if col not in self.df.columns:
            return [[] for _ in range(len(self.df))]
        return [
            [s.strip() for s in str(row).split(";") if s.strip()]
            for row in self.df[col]
        ]

    @staticmethod
    def synthetic(n_postings: int = 100) -> "JobPostingsDataset":
        """Tiny synthetic dataset for smoke testing."""
        import numpy as np
        rng = np.random.default_rng(0)

        titles = ["Data Analyst", "Software Engineer", "Product Manager",
                  "Economist", "UX Designer"]
        skills = [
            "Python; SQL; Statistics",
            "Python; Java; Docker; Kubernetes",
            "Agile; Roadmapping; Stakeholder Management",
            "Econometrics; STATA; R; Policy Analysis",
            "Figma; User Research; Prototyping",
        ]
        rows = {
            "job_title":        [titles[rng.integers(5)] for _ in range(n_postings)],
            "description":      ["Sample job description."] * n_postings,
            "skills_required":  [skills[rng.integers(5)] for _ in range(n_postings)],
        }
        ds          = object.__new__(JobPostingsDataset)
        ds.df       = pd.DataFrame(rows)
        ds.esco     = None
        return ds
