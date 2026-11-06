"""DuckDB evidence storage and lookup."""

from __future__ import annotations

import duckdb
import argparse
import os
from pathlib import Path
import uuid
import pandas as pd
import data
import settings

TABLE_KEYS = {
    'eligible_months': ['month'],
    'analogs_by_month': ['month', 'analog_rank'],
    'cases': ['case_id'],
    'feature_metadata': ['feature'],
    'evidence_registry': ['evidence_id'],
    'relations': ['evidence_id'],
    'analogs': ['case_id', 'analog_rank'],
    'analog_summary': ['case_id'],
    'features_all30': ['case_id', 'feature'],
    'evidence': ['case_id', 'feature'],
}


# Evidence lookup
def connect(seed: int = 42) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(settings.database_path(seed)), read_only=True)


def rows(con: duckdb.DuckDBPyConnection, query: str, parameters: list | None = None) -> list[dict]:
    result = con.execute(query, parameters or [])
    columns = [column[0] for column in result.description]
    return [settings.canonical_names(dict(zip(columns, row))) for row in result.fetchall()]


class CaseEvidence:
    """Cached evidence rows for one case across five conditions."""

    def __init__(self, con: duckdb.DuckDBPyConnection, case_id: str) -> None:
        self.case_id = case_id
        cases = rows(con, 'SELECT * FROM cases WHERE case_id = ?', [case_id])
        if len(cases) != 1:
            raise KeyError(f'Expected one DB case: {case_id}')
        self.case = cases[0]
        selected = rows(
            con, 'SELECT * FROM evidence WHERE case_id = ? ORDER BY d_position', [case_id]
        )
        if not selected:
            raise ValueError(f'Missing feature evidence: {case_id}')
        self.features = {row['feature']: row for row in selected}
        self.relations = rows(con, 'SELECT * FROM relations WHERE case_id = ?', [case_id])
        analogs = rows(
            con, 'SELECT * FROM analogs WHERE case_id = ? ORDER BY analog_rank', [case_id]
        )
        self.analogs = {row['analog_rank']: row for row in analogs}
        summaries = rows(con, 'SELECT * FROM analog_summary WHERE case_id = ?', [case_id])
        if len(summaries) != 1:
            raise ValueError(f'Missing analog summary: {case_id}')
        self.summary = summaries[0]
        registry = rows(con, 'SELECT * FROM evidence_registry WHERE case_id = ?', [case_id])
        self.registry = {row['evidence_id']: row for row in registry}

    def visible_ids(self, condition: str) -> set[str]:
        return {eid for eid, row in self.registry.items() if condition in row['visible_in']}

    def focal_features(self) -> list[str]:
        largest = max((row['phi_abs'] for row in self.features.values()))
        return [feature for feature, row in self.features.items() if row['phi_abs'] == largest]

    def relation_pairs(self) -> set[tuple[str, str]]:
        return {(row['overtaker'], row['overtaken']) for row in self.relations}

    def cited(
        self, evidence_ids: list[str], types: tuple[str, ...], feature: str | None = None
    ) -> dict | None:
        """Citation-order lookup of original evidence rows."""
        feature_columns = {
            'LOCAL_FEATURE': 'eid_loc',
            'GLOBAL_FEATURE': 'eid_glb',
            'TEMPORAL_SEQUENCE': 'eid_seq',
            'TEMPORAL_NUMERICAL': 'eid_tmp',
        }
        for eid in evidence_ids:
            record = self.registry[eid]
            evidence_type = record['type']
            if evidence_type not in types:
                continue
            if evidence_type in feature_columns:
                column = feature_columns[evidence_type]
                candidates = (
                    [self.features[feature]] if feature in self.features else self.features.values()
                )
                for row in candidates:
                    if row[column] == eid and (feature is None or row['feature'] == feature):
                        return row
            elif evidence_type == 'PREDICTION':
                return self.case
            elif evidence_type == 'OVERTAKES':
                return next((row for row in self.relations if row['evidence_id'] == eid))
            elif evidence_type == 'ANALOG_SUMMARY':
                return self.summary
            elif evidence_type in (
                'HISTORICAL_ANALOG',
                'SIMILAR_TO',
                'ANALOG_UNIVERSE_OUTCOME',
                'ANALOG_MODEL_PERFORMANCE',
            ):
                return self.analogs[int(eid.rsplit(':', 1)[1])]
        return None


# Database validation
def validate_database(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Table keys, cardinalities, visibility, and evidence references."""
    counts = {}
    for table, keys in TABLE_KEYS.items():
        counts[table] = con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
        columns = ', '.join(keys)
        duplicates = con.execute(
            f'SELECT count(*) FROM (SELECT {columns} FROM {table} '
            f'GROUP BY {columns} HAVING count(*) > 1)'
        ).fetchone()[0]
        if duplicates:
            raise ValueError(f'Duplicate keys in {table}')
        nulls = ' OR '.join((f'{column} IS NULL' for column in keys))
        if con.execute(f'SELECT count(*) FROM {table} WHERE {nulls}').fetchone()[0]:
            raise ValueError(f'Null key in {table}')
    if not counts['cases'] or counts['feature_metadata'] != 30:
        raise ValueError('Expected cases and exactly 30 feature definitions')
    for table, expected in [('analogs', '5'), ('analog_summary', '1'), ('features_all30', '30')]:
        bad = con.execute(
            f'SELECT count(*) FROM cases c LEFT JOIN '
            f'(SELECT case_id, count(*) n FROM {table} GROUP BY case_id) '
            f't USING (case_id) WHERE coalesce(t.n, 0) <> {expected}'
        ).fetchone()[0]
        if bad:
            raise ValueError(f'Invalid per-case row count in {table}')
    bad = con.execute(
        (
            'SELECT count(*) FROM cases c LEFT JOIN (SELECT case_id, count(*) n'
            ' FROM evidence GROUP BY case_id) e USING (case_id) WHERE '
            'coalesce(e.n, 0) <> c.n_display OR e.n NOT BETWEEN 3 AND 6'
        )
    ).fetchone()[0]
    if bad:
        raise ValueError('Invalid selected-feature count')
    for layer, visible in settings.VISIBILITY.items():
        if con.execute(
            'SELECT count(*) FROM evidence_registry WHERE layer = ? AND visible_in <> ?',
            [layer, visible],
        ).fetchone()[0]:
            raise ValueError(f'Invalid condition visibility: {layer}')
    for table in (
        'evidence',
        'evidence_registry',
        'relations',
        'analogs',
        'analog_summary',
        'features_all30',
    ):
        if con.execute(
            f'SELECT count(*) FROM {table} t ANTI JOIN cases c USING (case_id)'
        ).fetchone()[0]:
            raise ValueError(f'Unknown case in {table}')
    for field in ('eid_loc', 'eid_glb', 'eid_seq', 'eid_tmp'):
        if con.execute(
            f'SELECT count(*) FROM evidence e LEFT JOIN evidence_registry r '
            f'ON e.{field} = r.evidence_id AND e.case_id = r.case_id '
            f'WHERE r.evidence_id IS NULL'
        ).fetchone()[0]:
            raise ValueError(f'Missing registered identifier: {field}')
    return counts


RO = {feature: index for index, feature in enumerate(data.RETRIEVAL_ORDER)}


# Database storage
def write_database(tables: dict[str, pd.DataFrame], path: Path) -> dict[str, int]:
    if set(tables) != set(TABLE_KEYS):
        raise ValueError('The evidence database must contain the ten specified tables.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.stem}-{uuid.uuid4().hex}.duckdb')
    con = None
    try:
        con = duckdb.connect(str(temporary))
        con.execute('BEGIN TRANSACTION')
        for table, frame in tables.items():
            if table == 'relations' and frame.empty:
                con.execute(
                    (
                        'CREATE TABLE relations (case_id VARCHAR, month VARCHAR, '
                        'evidence_id VARCHAR, overtaker VARCHAR, overtaken VARCHAR, '
                        'start_P_overtaker DOUBLE, start_P_overtaken DOUBLE, '
                        'end_P_overtaker DOUBLE, end_P_overtaken DOUBLE)'
                    )
                )
            else:
                con.register('_table_input', frame)
                con.execute(f'CREATE TABLE {table} AS SELECT * FROM _table_input')
                con.unregister('_table_input')
            keys = ', '.join(TABLE_KEYS[table])
            con.execute(f'ALTER TABLE {table} ADD PRIMARY KEY ({keys})')
        con.execute('CREATE INDEX evidence_case ON evidence(case_id)')
        con.execute('CREATE INDEX registry_case ON evidence_registry(case_id)')
        counts = validate_database(con)
        con.execute('COMMIT')
        con.close()
        con = None
        os.replace(temporary, path)
        return counts
    finally:
        if con is not None:
            con.close()
        if temporary.exists():
            temporary.unlink()


def feature_metadata_table() -> pd.DataFrame:
    """Feature definitions stored in the DB for prompt rendering."""
    meta = settings.feature_metadata()['features']
    rows = []
    for f in data.RETRIEVAL_ORDER:
        m = meta[f]
        rows.append(
            {
                'feature': f,
                'retrieval_order': RO[f],
                'name_en': m['name_en'],
                'formula_text': m.get('formula_en') or m['formula'],
                'window_months': m.get('window_months'),
                'higher_means_en': m['higher_means_en'],
                'discrete': bool(m.get('discrete')),
                'caveat_en': m.get('caveat_en'),
            }
        )
    return pd.DataFrame(rows)
