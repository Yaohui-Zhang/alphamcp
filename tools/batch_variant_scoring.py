"""
Batch variant scoring tutorial demonstrating how to score multiple genetic variants using AlphaGenome.

This MCP Server provides 2 tools:
1. score_variants_batch: Score multiple genetic variants in batch using configurable variant scorers
2. filter_variant_scores: Filter variant scores by ontology criteria (e.g., cell types)

All tools extracted from `alphagenome/colabs/batch_variant_scoring.ipynb`.
"""

# Standard imports
from typing import Annotated, Literal, Any
import pandas as pd
import numpy as np
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime
from io import StringIO

# AlphaGenome imports
from alphagenome import colab_utils
from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers
from tqdm import tqdm

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp_inputs" / "batch_variant_scoring"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp_outputs" / "batch_variant_scoring"

INPUT_DIR = Path(os.environ.get("BATCH_VARIANT_SCORING_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("BATCH_VARIANT_SCORING_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
batch_variant_scoring_mcp = FastMCP(name="batch_variant_scoring")

@batch_variant_scoring_mcp.tool
def score_variants_batch(
    # Primary data inputs
    vcf_path: Annotated[str | None, "Path to VCF file with extension .vcf or .tsv. The header of the file should include the following columns: variant_id, CHROM, POS, REF, ALT"] = None,
    # Analysis parameters with tutorial defaults
    api_key: Annotated[str, "AlphaGenome API key for authentication"] = "",
    organism: Annotated[Literal["human", "mouse"], "Organism for variant scoring"] = "human",
    sequence_length: Annotated[Literal["2KB", "16KB", "100KB", "500KB", "1MB"], "Length of sequence around variants to predict"] = "1MB",
    score_rna_seq: Annotated[bool, "Score RNA-seq effects"] = True,
    score_cage: Annotated[bool, "Score CAGE effects"] = True,
    score_procap: Annotated[bool, "Score ProCAP effects"] = True,
    score_atac: Annotated[bool, "Score ATAC-seq effects"] = True,
    score_dnase: Annotated[bool, "Score DNase effects"] = True,
    score_chip_histone: Annotated[bool, "Score ChIP histone effects"] = True,
    score_chip_tf: Annotated[bool, "Score ChIP transcription factor effects"] = True,
    score_polyadenylation: Annotated[bool, "Score polyadenylation effects"] = True,
    score_splice_sites: Annotated[bool, "Score splice sites effects"] = True,
    score_splice_site_usage: Annotated[bool, "Score splice site usage effects"] = True,
    score_splice_junctions: Annotated[bool, "Score splice junctions effects"] = True,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Score multiple genetic variants in batch using AlphaGenome with configurable variant scorers.
    Input is VCF file with variant information and output is comprehensive variant scores table.
    """
    # Validate exactly one input
    if vcf_path is None:
        raise ValueError("Path to VCF file must be provided")
    
    # Set output prefix
    if out_prefix is None:
        out_prefix = f"batch_variant_scores_{timestamp}"
    
    # Load VCF file containing variants
    vcf = pd.read_csv(vcf_path, sep='\t')
    
    # Validate required columns
    required_columns = ['variant_id', 'CHROM', 'POS', 'REF', 'ALT']
    for column in required_columns:
        if column not in vcf.columns:
            raise ValueError(f'VCF file is missing required column: {column}.')
    
    # Load the model
    dna_model = dna_client.create(api_key)
    
    # Parse organism specification
    organism_map = {
        'human': dna_client.Organism.HOMO_SAPIENS,
        'mouse': dna_client.Organism.MUS_MUSCULUS,
    }
    organism_enum = organism_map[organism]
    
    # Parse sequence length
    sequence_length_value = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f'SEQUENCE_LENGTH_{sequence_length}'
    ]
    
    # Parse scorer specification
    scorer_selections = {
        'rna_seq': score_rna_seq,
        'cage': score_cage,
        'procap': score_procap,
        'atac': score_atac,
        'dnase': score_dnase,
        'chip_histone': score_chip_histone,
        'chip_tf': score_chip_tf,
        'polyadenylation': score_polyadenylation,
        'splice_sites': score_splice_sites,
        'splice_site_usage': score_splice_site_usage,
        'splice_junctions': score_splice_junctions,
    }
    
    all_scorers = variant_scorers.RECOMMENDED_VARIANT_SCORERS
    selected_scorers = [
        all_scorers[key]
        for key in all_scorers
        if scorer_selections.get(key.lower(), False)
    ]
    
    # Remove any scorers or output types that are not supported for the chosen organism
    unsupported_scorers = [
        scorer
        for scorer in selected_scorers
        if (
            organism_enum.value
            not in variant_scorers.SUPPORTED_ORGANISMS[scorer.base_variant_scorer]
        )
        | (
            (scorer.requested_output == dna_client.OutputType.PROCAP)
            & (organism_enum == dna_client.Organism.MUS_MUSCULUS)
        )
    ]
    if len(unsupported_scorers) > 0:
        print(
            f'Excluding {unsupported_scorers} scorers as they are not supported for'
            f' {organism_enum}.'
        )
        for unsupported_scorer in unsupported_scorers:
            selected_scorers.remove(unsupported_scorer)
    
    # Score variants in the VCF file
    results = []
    
    for i, vcf_row in tqdm(vcf.iterrows(), total=len(vcf)):
        variant = genome.Variant(
            chromosome=str(vcf_row.CHROM),
            position=int(vcf_row.POS),
            reference_bases=vcf_row.REF,
            alternate_bases=vcf_row.ALT,
            name=vcf_row.variant_id,
        )
        interval = variant.reference_interval.resize(sequence_length_value)
        
        variant_scores = dna_model.score_variant(
            interval=interval,
            variant=variant,
            variant_scorers=selected_scorers,
            organism=organism_enum,
        )
        results.append(variant_scores)
    
    df_scores = variant_scorers.tidy_scores(results)
    
    # Save results
    output_file = OUTPUT_DIR / f"{out_prefix}.csv"
    df_scores.to_csv(output_file, index=False)
    
    return {
        "message": f"Batch variant scoring completed for {len(vcf)} variants with {len(selected_scorers)} scorers",
        "reference": "alphagenome/colabs/batch_variant_scoring.ipynb",
        "artifacts": [
            {
                "description": "Batch variant scores results",
                "path": str(output_file.resolve())
            }
        ]
    }

@batch_variant_scoring_mcp.tool
def filter_variant_scores(
    # Primary data inputs
    scores_path: Annotated[str | None, "Path to variant scores CSV file from batch scoring"] = None,
    # Analysis parameters with tutorial defaults
    ontology_curie: Annotated[str, "Ontology CURIE for filtering (e.g., 'CL:0000084' for T-cells)"] = "CL:0000084",
    exclude_ontology_column: Annotated[bool, "Whether to exclude ontology_curie column from output"] = True,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Filter variant scores by ontology criteria to examine effects on specific cell types or tissues.
    Input is variant scores CSV file and output is filtered scores table.
    """
    # Validate exactly one input
    if scores_path is None:
        raise ValueError("Path to variant scores CSV file must be provided")
    
    # Set output prefix
    if out_prefix is None:
        out_prefix = f"filtered_variant_scores_{timestamp}"
    
    # Load variant scores
    df_scores = pd.read_csv(scores_path)
    
    # Filter by ontology criteria
    filtered_df = df_scores[df_scores['ontology_curie'] == ontology_curie]
    
    # Optionally exclude ontology column
    if exclude_ontology_column:
        columns = [c for c in filtered_df.columns if c != 'ontology_curie']
        filtered_df = filtered_df[columns]
    
    # Save filtered results
    output_file = OUTPUT_DIR / f"{out_prefix}.csv"
    filtered_df.to_csv(output_file, index=False)
    
    return {
        "message": f"Filtered {len(filtered_df)} variant scores for ontology {ontology_curie}",
        "reference": "alphagenome/colabs/batch_variant_scoring.ipynb",
        "artifacts": [
            {
                "description": "Filtered variant scores",
                "path": str(output_file.resolve())
            }
        ]
    }