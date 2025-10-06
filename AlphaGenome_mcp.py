"""
Model Context Protocol (MCP) for AlphaGenome

AlphaGenome provides state-of-the-art AI-driven tools for genomic sequence analysis, variant effect prediction, and functional genomics visualization. The platform enables researchers to predict regulatory elements, assess variant impacts, and explore molecular mechanisms across diverse cell types and tissues.

This MCP Server contains the tools extracted from the following tutorials with their tools:
1. batch_variant_scoring
    - score_variants_batch: Score multiple genetic variants in batch using configurable variant scorers
    - filter_variant_scores: Filter variant scores by ontology criteria (e.g., cell types)
2. essential_commands
    - create_genomic_interval: Create genomic intervals for DNA regions
    - create_genomic_variant: Create genomic variants for genetic changes  
    - create_track_data: Create TrackData objects from arrays and metadata
    - create_variant_scores: Create AnnData objects for variant scoring results
    - genomic_interval_operations: Perform operations on genomic intervals
    - variant_interval_operations: Check variant overlaps with intervals  
    - track_data_operations: Filter, resize, slice and transform TrackData
    - track_data_resolution_conversion: Convert between track data resolutions
3. example_analysis_workflow
    # REMOVED - Too TAL1-specific:
    # - visualize_tal1_variant_positions: Visualize genomic positions of oncogenic TAL1 variants
    # - predict_variant_functional_impact: Predict functional effects of specific TAL1 variants
    # - compare_oncogenic_vs_background_variants: Compare predicted effects of disease variants vs background
4. quick_start
    - predict_dna_sequence: Predict genomic tracks from DNA sequence
    - predict_genome_interval: Predict genomic tracks for reference genome intervals  
    # REMOVED - Redundant with visualize_variant_effects:
    # - predict_variant_effects: Predict and visualize genetic variant effects
    # REMOVED - Redundant with score_variants_batch:
    # - score_variant_effect: Score genetic variant effects using variant scorers
    - ism_analysis: Perform in silico mutagenesis analysis with sequence logos
    # REMOVED - Should be parameter in other functions:
    # - mouse_predictions: Make predictions for mouse sequences and intervals
5. tissue_ontology_mapping
    - explore_output_metadata: Explore and filter output metadata for specific organisms and search terms
    - count_tracks_by_output_type: Count tracks by output type for human and mouse organisms
6. variant_scoring_ui
    # REMOVED - Redundant with score_variants_batch:
    # - score_variant: Score a single variant with multiple variant scorers and save results
    - visualize_variant_effects: Generate comprehensive variant effect visualization across multiple modalities
7. visualization_modality_tour
    - visualize_gene_expression: Visualize RNA_SEQ and CAGE gene expression predictions
    # REMOVED - Redundant with visualize_variant_effects:
    # - visualize_variant_expression_effects: Show REF vs ALT variant effects on gene expression
    # REMOVED - Too specific:
    # - visualize_custom_annotations: Plot custom annotations like polyadenylation sites
    - visualize_chromatin_accessibility: Visualize DNASE and ATAC chromatin accessibility
    - visualize_splicing_effects: Visualize splicing predictions with SPLICE_SITES, SPLICE_SITE_USAGE, SPLICE_JUNCTIONS
    # REMOVED - Redundant with visualize_variant_effects:
    # - visualize_variant_splicing_effects: Show REF vs ALT variant effects on splicing with sashimi plots
    - visualize_histone_modifications: Visualize CHIP_HISTONE predictions with custom colors
    - visualize_tf_binding: Visualize CHIP_TF transcription factor binding predictions
    - visualize_contact_maps: Visualize CONTACT_MAPS DNA-DNA contact predictions
"""

import sys
from pathlib import Path
from fastmcp import FastMCP

# Import the MCP tools from the tools folder
from tools.batch_variant_scoring import batch_variant_scoring_mcp
from tools.essential_commands import essential_commands_mcp
# REMOVED - All tools are TAL1-specific:
# from tools.example_analysis_workflow import example_analysis_workflow_mcp
# from tools.quick_start import quick_start_mcp
# from tools.tissue_ontology_mapping import tissue_ontology_mapping_mcp
# from tools.variant_scoring_ui import variant_scoring_ui_mcp
# from tools.visualization_modality_tour import visualization_modality_tour_mcp

# Define the MCP server
mcp = FastMCP(name = "AlphaGenome")

# Mount the tools
mcp.mount(batch_variant_scoring_mcp)
mcp.mount(essential_commands_mcp)
# REMOVED - All tools are TAL1-specific:
# mcp.mount(example_analysis_workflow_mcp)
# mcp.mount(quick_start_mcp)
# mcp.mount(tissue_ontology_mapping_mcp)
# mcp.mount(variant_scoring_ui_mcp)
# mcp.mount(visualization_modality_tour_mcp)

# Run the MCP server
if __name__ == "__main__":
  mcp.run(show_banner=False)