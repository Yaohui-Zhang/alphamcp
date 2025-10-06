"""
Tissue ontology mapping tutorial for navigating biological data ontologies in AlphaGenome.

This MCP Server provides 2 tools:
1. explore_output_metadata: Explore and filter output metadata for specific organisms and search terms
2. count_tracks_by_output_type: Count tracks by output type for human and mouse organisms

All tools extracted from `alphagenome/colabs/tissue_ontology_mapping.ipynb`.
"""

# Standard imports
from typing import Annotated, Literal, Any
import pandas as pd
import numpy as np
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp_inputs" / "tissue_ontology_mapping"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp_outputs" / "tissue_ontology_mapping"

INPUT_DIR = Path(os.environ.get("TISSUE_ONTOLOGY_MAPPING_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("TISSUE_ONTOLOGY_MAPPING_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
tissue_ontology_mapping_mcp = FastMCP(name="tissue_ontology_mapping")

@tissue_ontology_mapping_mcp.tool
def explore_output_metadata(
    # Analysis parameters with tutorial defaults
    organism: Annotated[Literal["HOMO_SAPIENS", "MUS_MUSCULUS"], "Target organism for metadata exploration"] = "HOMO_SAPIENS",
    api_key: Annotated[str, "AlphaGenome API key for accessing the DNA model"] = "",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Explore output metadata for specific organisms to find ontology terms and tissue types.
    Input is organism selection and API key and output is metadata table for interactive exploration.
    """
    # Import required modules
    from alphagenome.models import dna_client
    
    if not api_key:
        raise ValueError("API key must be provided")
    
    # Create DNA model client
    dna_model = dna_client.create(api_key)
    
    # Get organism enum
    org_enum = getattr(dna_client.Organism, organism)
    
    # Get output metadata
    output_metadata = dna_model.output_metadata(org_enum).concatenate()
    
    # Set output filename
    if out_prefix is None:
        out_prefix = f"output_metadata_{organism.lower()}"
    
    output_file = OUTPUT_DIR / f"{out_prefix}_{timestamp}.csv"
    
    # Save metadata as CSV
    output_metadata.to_csv(output_file, index=False)
    
    # Return standardized format
    return {
        "message": f"Output metadata exploration completed for {organism}",
        "reference": "alphagenome/colabs/tissue_ontology_mapping.ipynb", 
        "artifacts": [
            {
                "description": f"Output metadata for {organism}",
                "path": str(output_file.resolve())
            }
        ]
    }

@tissue_ontology_mapping_mcp.tool
def count_tracks_by_output_type(
    # Analysis parameters with tutorial defaults
    api_key: Annotated[str, "AlphaGenome API key for accessing the DNA model"] = "",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Count tracks by output type for both human and mouse organisms to understand data availability.
    Input is API key and output is track counts table comparing human vs mouse availability.
    """
    # Import required modules
    from alphagenome.models import dna_client
    
    if not api_key:
        raise ValueError("API key must be provided")
    
    # Create DNA model client
    dna_model = dna_client.create(api_key)
    
    # Count human tracks
    human_tracks = (
        dna_model.output_metadata(dna_client.Organism.HOMO_SAPIENS)
        .concatenate()
        .groupby('output_type')
        .size()
        .rename('# Human tracks')
    )
    
    # Count mouse tracks
    mouse_tracks = (
        dna_model.output_metadata(dna_client.Organism.MUS_MUSCULUS)
        .concatenate()
        .groupby('output_type')
        .size()
        .rename('# Mouse tracks')
    )
    
    # Combine the results
    track_counts = pd.concat([human_tracks, mouse_tracks], axis=1).astype(pd.Int64Dtype())
    
    # Set output filename
    if out_prefix is None:
        out_prefix = "track_counts"
    
    output_file = OUTPUT_DIR / f"{out_prefix}_{timestamp}.csv"
    
    # Save track counts as CSV
    track_counts.to_csv(output_file)
    
    # Return standardized format
    return {
        "message": "Track counting by output type completed successfully",
        "reference": "alphagenome/colabs/tissue_ontology_mapping.ipynb",
        "artifacts": [
            {
                "description": "Track counts by output type",
                "path": str(output_file.resolve())
            }
        ]
    }