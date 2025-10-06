"""
AlphaGenome Quick Start tutorial tools for DNA sequence analysis and prediction.

This MCP Server provides 6 tools:
1. predict_dna_sequence: Predict genomic tracks from DNA sequence
2. predict_genome_interval: Predict genomic tracks for reference genome intervals  
3. predict_variant_effects: Predict and visualize genetic variant effects
4. score_variant_effect: Score genetic variant effects using variant scorers
5. ism_analysis: Perform in silico mutagenesis analysis with sequence logos
# REMOVED - Too specific:
# 6. mouse_predictions: Make predictions for mouse sequences and intervals

All tools extracted from `alphagenome/colabs/quick_start.ipynb`.
"""

# Standard imports
from typing import Annotated, Literal, Any
import pandas as pd
import numpy as np
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime

# AlphaGenome imports
from alphagenome import colab_utils
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp_inputs" / "quick_start"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp_outputs" / "quick_start"

INPUT_DIR = Path(os.environ.get("QUICK_START_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("QUICK_START_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
quick_start_mcp = FastMCP(name="quick_start")

@quick_start_mcp.tool
def predict_dna_sequence(
    api_key: Annotated[str, "AlphaGenome API key for authentication"],
    sequence: Annotated[str, "DNA sequence to analyze (will be center-padded to valid length)"] = 'GATTACA',
    sequence_length: Annotated[Literal["2KB", "16KB", "100KB", "500KB", "1MB"], "Model input sequence length"] = "2KB",
    output_types: Annotated[list[str], "List of output types to predict (e.g. ['DNASE', 'CAGE', 'RNA_SEQ'])"] = ["DNASE"],
    ontology_terms: Annotated[list[str], "List of ontology terms for tissues/cell types (e.g. ['UBERON:0002048'])"] = ["UBERON:0002048"],
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Predict genomic tracks from a DNA sequence using AlphaGenome model.
    Input is DNA sequence string and output is track predictions with metadata tables.
    """
    # Set up output prefix
    if out_prefix is None:
        out_prefix = f"predict_dna_sequence_{timestamp}"
    
    # Create DNA model
    dna_model = dna_client.create(api_key)
    
    # Convert sequence length to client constant
    length_map = {
        "2KB": dna_client.SEQUENCE_LENGTH_2KB,
        "16KB": dna_client.SEQUENCE_LENGTH_16KB, 
        "100KB": dna_client.SEQUENCE_LENGTH_100KB,
        "500KB": dna_client.SEQUENCE_LENGTH_500KB,
        "1MB": dna_client.SEQUENCE_LENGTH_1MB
    }
    target_length = length_map[sequence_length]
    
    # Pad sequence to valid length
    padded_sequence = sequence.center(target_length, 'N')
    
    # Convert output types to client enums
    output_enums = []
    for output_type in output_types:
        output_enums.append(getattr(dna_client.OutputType, output_type))
    
    # Make prediction
    output = dna_model.predict_sequence(
        sequence=padded_sequence,
        requested_outputs=output_enums,
        ontology_terms=ontology_terms,
    )
    
    artifacts = []
    
    # Save track data and metadata for each output type
    for output_type in output_types:
        track_data = getattr(output, output_type.lower())
        
        # Save values
        values_file = OUTPUT_DIR / f"{out_prefix}_{output_type.lower()}_values.csv"
        pd.DataFrame(track_data.values).to_csv(values_file, index=False)
        artifacts.append({
            "description": f"{output_type} prediction values",
            "path": str(values_file.resolve())
        })
        
        # Save metadata
        metadata_file = OUTPUT_DIR / f"{out_prefix}_{output_type.lower()}_metadata.csv"
        track_data.metadata.to_csv(metadata_file, index=False)
        artifacts.append({
            "description": f"{output_type} track metadata", 
            "path": str(metadata_file.resolve())
        })
    
    return {
        "message": f"DNA sequence predictions completed for {len(output_types)} output types",
        "reference": "alphagenome/colabs/quick_start.ipynb",
        "artifacts": artifacts
    }

@quick_start_mcp.tool
def predict_genome_interval(
    api_key: Annotated[str, "AlphaGenome API key for authentication"],
    chromosome: Annotated[str, "Chromosome name (e.g. 'chr19')"] = "chr19",
    start_position: Annotated[int, "Start position on chromosome"] = 40991281,
    end_position: Annotated[int, "End position on chromosome"] = 41018398,
    strand: Annotated[Literal["+", "-", "."], "Strand orientation"] = "+",
    sequence_length: Annotated[Literal["2KB", "16KB", "100KB", "500KB", "1MB"], "Model input sequence length"] = "1MB",
    output_types: Annotated[list[str], "List of output types to predict (e.g. ['RNA_SEQ'])"] = ["RNA_SEQ"],
    ontology_terms: Annotated[list[str], "List of ontology terms for tissues/cell types"] = ["UBERON:0001114"],
    gene_symbol: Annotated[str | None, "Gene symbol to center interval on (overrides coordinates)"] = "CYP2B6",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Predict genomic tracks for a reference genome interval with transcript visualization.
    Input is genomic coordinates or gene symbol and output is prediction plot and metadata.
    """
    # Set up output prefix
    if out_prefix is None:
        out_prefix = f"predict_genome_interval_{timestamp}"
        
    # Create DNA model
    dna_model = dna_client.create(api_key)
    
    # Load GTF file for gene annotation
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    
    # Set up transcript extractors
    gtf_transcripts = gene_annotation.filter_protein_coding(gtf)
    gtf_transcripts = gene_annotation.filter_to_longest_transcript(gtf_transcripts)
    transcript_extractor = transcript_utils.TranscriptExtractor(gtf_transcripts)
    
    # Create interval - use gene symbol if provided, otherwise coordinates
    if gene_symbol:
        interval = gene_annotation.get_gene_interval(gtf, gene_symbol=gene_symbol)
    else:
        interval = genome.Interval(chromosome, start_position, end_position, strand)
    
    # Resize to model-compatible length
    length_map = {
        "2KB": dna_client.SEQUENCE_LENGTH_2KB,
        "16KB": dna_client.SEQUENCE_LENGTH_16KB,
        "100KB": dna_client.SEQUENCE_LENGTH_100KB,
        "500KB": dna_client.SEQUENCE_LENGTH_500KB,
        "1MB": dna_client.SEQUENCE_LENGTH_1MB
    }
    interval = interval.resize(length_map[sequence_length])
    
    # Convert output types to client enums
    output_enums = []
    for output_type in output_types:
        output_enums.append(getattr(dna_client.OutputType, output_type))
    
    # Make prediction
    output = dna_model.predict_interval(
        interval=interval,
        requested_outputs=output_enums,
        ontology_terms=ontology_terms,
    )
    
    # Extract transcripts for visualization
    longest_transcripts = transcript_extractor.extract(interval)
    
    artifacts = []
    
    # Save metadata for each output type
    for output_type in output_types:
        track_data = getattr(output, output_type.lower())
        
        # Save metadata
        metadata_file = OUTPUT_DIR / f"{out_prefix}_{output_type.lower()}_metadata.csv"
        track_data.metadata.to_csv(metadata_file, index=False)
        artifacts.append({
            "description": f"{output_type} track metadata",
            "path": str(metadata_file.resolve())
        })
    
    # Create visualization - full interval
    plt.figure(figsize=(15, 8))
    track_data = getattr(output, output_types[0].lower())
    plot_components.plot(
        components=[
            plot_components.TranscriptAnnotation(longest_transcripts),
            plot_components.Tracks(track_data),
        ],
        interval=track_data.interval,
    )
    
    plot_file = OUTPUT_DIR / f"{out_prefix}_{output_types[0].lower()}_plot.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts.append({
        "description": f"{output_types[0]} prediction plot",
        "path": str(plot_file.resolve())
    })
    
    # Create zoomed visualization
    plt.figure(figsize=(15, 8))
    plot_components.plot(
        components=[
            plot_components.TranscriptAnnotation(longest_transcripts, fig_height=0.1),
            plot_components.Tracks(track_data),
        ],
        interval=track_data.interval.resize(2**15),
    )
    
    plot_zoomed_file = OUTPUT_DIR / f"{out_prefix}_{output_types[0].lower()}_plot_zoomed.png"
    plt.savefig(plot_zoomed_file, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts.append({
        "description": f"{output_types[0]} prediction plot (zoomed)",
        "path": str(plot_zoomed_file.resolve())
    })
    
    return {
        "message": f"Genome interval predictions completed with {len(longest_transcripts)} transcripts",
        "reference": "alphagenome/colabs/quick_start.ipynb", 
        "artifacts": artifacts
    }

@quick_start_mcp.tool 
def predict_variant_effects(
    api_key: Annotated[str, "AlphaGenome API key for authentication"],
    chromosome: Annotated[str, "Chromosome name (e.g. 'chr22')"] = "chr22",
    position: Annotated[int, "Variant position"] = 36201698,
    reference_bases: Annotated[str, "Reference allele"] = "A", 
    alternate_bases: Annotated[str, "Alternative allele"] = "C",
    sequence_length: Annotated[Literal["2KB", "16KB", "100KB", "500KB", "1MB"], "Model input sequence length"] = "1MB",
    output_types: Annotated[list[str], "List of output types to predict"] = ["RNA_SEQ"],
    ontology_terms: Annotated[list[str], "List of ontology terms for tissues/cell types"] = ["UBERON:0001157"],
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Predict and visualize genetic variant effects comparing REF vs ALT predictions.
    Input is variant coordinates and output is overlaid REF/ALT visualization plot.
    """
    # Set up output prefix
    if out_prefix is None:
        out_prefix = f"predict_variant_effects_{timestamp}"
        
    # Create DNA model
    dna_model = dna_client.create(api_key)
    
    # Create variant object
    variant = genome.Variant(
        chromosome=chromosome,
        position=position,
        reference_bases=reference_bases,
        alternate_bases=alternate_bases,
    )
    
    # Create interval from variant
    length_map = {
        "2KB": dna_client.SEQUENCE_LENGTH_2KB,
        "16KB": dna_client.SEQUENCE_LENGTH_16KB,
        "100KB": dna_client.SEQUENCE_LENGTH_100KB,
        "500KB": dna_client.SEQUENCE_LENGTH_500KB,
        "1MB": dna_client.SEQUENCE_LENGTH_1MB
    }
    interval = variant.reference_interval.resize(length_map[sequence_length])
    
    # Convert output types to client enums
    output_enums = []
    for output_type in output_types:
        output_enums.append(getattr(dna_client.OutputType, output_type))
    
    # Make variant prediction
    variant_output = dna_model.predict_variant(
        interval=interval,
        variant=variant,
        requested_outputs=output_enums,
        ontology_terms=ontology_terms,
    )
    
    # Load GTF for transcript annotation
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    gtf_transcripts = gene_annotation.filter_protein_coding(gtf)
    gtf_transcripts = gene_annotation.filter_to_longest_transcript(gtf_transcripts)
    transcript_extractor = transcript_utils.TranscriptExtractor(gtf_transcripts)
    longest_transcripts = transcript_extractor.extract(interval)
    
    artifacts = []
    
    # Create overlaid REF vs ALT visualization
    plt.figure(figsize=(15, 8))
    ref_track_data = getattr(variant_output.reference, output_types[0].lower())
    alt_track_data = getattr(variant_output.alternate, output_types[0].lower())
    
    plot_components.plot(
        [
            plot_components.TranscriptAnnotation(longest_transcripts),
            plot_components.OverlaidTracks(
                tdata={
                    'REF': ref_track_data,
                    'ALT': alt_track_data,
                },
                colors={'REF': 'dimgrey', 'ALT': 'red'},
            ),
        ],
        interval=ref_track_data.interval.resize(2**15),
        # Annotate the location of the variant as a vertical line
        annotations=[plot_components.VariantAnnotation([variant], alpha=0.8)],
    )
    
    plot_file = OUTPUT_DIR / f"{out_prefix}_{output_types[0].lower()}_variant_plot.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts.append({
        "description": f"{output_types[0]} REF vs ALT comparison plot",
        "path": str(plot_file.resolve())
    })
    
    return {
        "message": f"Variant effect predictions completed for {chromosome}:{position}:{reference_bases}>{alternate_bases}",
        "reference": "alphagenome/colabs/quick_start.ipynb",
        "artifacts": artifacts
    }

# @quick_start_mcp.tool
def score_variant_effect(
    api_key: Annotated[str, "AlphaGenome API key for authentication"],
    chromosome: Annotated[str, "Chromosome name (e.g. 'chr22')"] = "chr22",
    position: Annotated[int, "Variant position"] = 36201698,
    reference_bases: Annotated[str, "Reference allele"] = "A",
    alternate_bases: Annotated[str, "Alternative allele"] = "C", 
    sequence_length: Annotated[Literal["2KB", "16KB", "100KB", "500KB", "1MB"], "Model input sequence length"] = "1MB",
    scorer_type: Annotated[Literal["RNA_SEQ", "DNASE", "CAGE", "ATAC"], "Variant scorer type to use"] = "RNA_SEQ",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Score genetic variant effects using recommended variant scorers and produce tidy scores.
    Input is variant coordinates and output is variant scores table with genes and tracks.
    """
    # Set up output prefix
    if out_prefix is None:
        out_prefix = f"score_variant_effect_{timestamp}"
        
    # Create DNA model
    dna_model = dna_client.create(api_key)
    
    # Create variant object
    variant = genome.Variant(
        chromosome=chromosome,
        position=position,
        reference_bases=reference_bases,
        alternate_bases=alternate_bases,
    )
    
    # Create interval from variant
    length_map = {
        "2KB": dna_client.SEQUENCE_LENGTH_2KB,
        "16KB": dna_client.SEQUENCE_LENGTH_16KB,
        "100KB": dna_client.SEQUENCE_LENGTH_100KB,
        "500KB": dna_client.SEQUENCE_LENGTH_500KB,
        "1MB": dna_client.SEQUENCE_LENGTH_1MB
    }
    interval = variant.reference_interval.resize(length_map[sequence_length])
    
    # Get recommended variant scorer
    variant_scorer = variant_scorers.RECOMMENDED_VARIANT_SCORERS[scorer_type]
    
    # Score variant
    variant_scores = dna_model.score_variant(
        interval=interval,
        variant=variant, 
        variant_scorers=[variant_scorer]
    )
    
    artifacts = []
    
    # Extract first scorer results
    scores_adata = variant_scores[0]
    
    # Save gene metadata (obs)
    genes_file = OUTPUT_DIR / f"{out_prefix}_{scorer_type}_genes.csv"
    scores_adata.obs.to_csv(genes_file, index=True)
    artifacts.append({
        "description": f"{scorer_type} gene metadata",
        "path": str(genes_file.resolve())
    })
    
    # Save track metadata (var)
    tracks_file = OUTPUT_DIR / f"{out_prefix}_{scorer_type}_tracks.csv" 
    scores_adata.var.to_csv(tracks_file, index=True)
    artifacts.append({
        "description": f"{scorer_type} track metadata",
        "path": str(tracks_file.resolve())
    })
    
    # Save raw scores matrix
    raw_scores_file = OUTPUT_DIR / f"{out_prefix}_{scorer_type}_raw_scores.csv"
    pd.DataFrame(scores_adata.X, 
                 index=scores_adata.obs.index, 
                 columns=scores_adata.var.index).to_csv(raw_scores_file, index=True)
    artifacts.append({
        "description": f"{scorer_type} raw scores matrix",
        "path": str(raw_scores_file.resolve())
    })
    
    # Create tidy scores dataframe
    tidy_scores_df = variant_scorers.tidy_scores([scores_adata], match_gene_strand=True)
    
    # Save tidy scores
    tidy_file = OUTPUT_DIR / f"{out_prefix}_{scorer_type}_tidy_scores.csv"
    tidy_scores_df.to_csv(tidy_file, index=False)
    artifacts.append({
        "description": f"{scorer_type} tidy scores table",
        "path": str(tidy_file.resolve())
    })
    
    return {
        "message": f"Variant scoring completed: {scores_adata.X.shape[0]} genes × {scores_adata.X.shape[1]} tracks",
        "reference": "alphagenome/colabs/quick_start.ipynb",
        "artifacts": artifacts
    }

@quick_start_mcp.tool
def ism_analysis(
    api_key: Annotated[str, "AlphaGenome API key for authentication"],
    chromosome: Annotated[str, "Chromosome for ISM analysis"] = "chr20",
    start_position: Annotated[int, "Start position for sequence context"] = 3753000,
    end_position: Annotated[int, "End position for sequence context"] = 3753400,
    sequence_length: Annotated[Literal["2KB", "16KB", "100KB", "500KB", "1MB"], "Model input sequence length"] = "2KB",
    ism_width: Annotated[int, "Width of region to mutate systematically"] = 256,
    output_type: Annotated[Literal["DNASE", "RNA_SEQ", "CAGE", "ATAC"], "Output type for scoring variants"] = "DNASE",
    mask_width: Annotated[int, "Width of center mask for scoring"] = 501,
    target_cell_line: Annotated[str, "Ontology term for specific cell line/tissue"] = "EFO:0002067",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Perform in silico mutagenesis analysis with sequence logo visualization of important regions.
    Input is genomic coordinates and output is ISM matrix and sequence logo plot.
    """
    # Set up output prefix
    if out_prefix is None:
        out_prefix = f"ism_analysis_{timestamp}"
        
    # Create DNA model
    dna_model = dna_client.create(api_key)
    
    # Create sequence interval
    sequence_interval = genome.Interval(chromosome, start_position, end_position)
    length_map = {
        "2KB": dna_client.SEQUENCE_LENGTH_2KB,
        "16KB": dna_client.SEQUENCE_LENGTH_16KB,
        "100KB": dna_client.SEQUENCE_LENGTH_100KB,
        "500KB": dna_client.SEQUENCE_LENGTH_500KB,
        "1MB": dna_client.SEQUENCE_LENGTH_1MB
    }
    sequence_interval = sequence_interval.resize(length_map[sequence_length])
    
    # Create ISM interval (region to mutate)
    ism_interval = sequence_interval.resize(ism_width)
    
    # Create variant scorer
    output_enum = getattr(dna_client.OutputType, output_type)
    variant_scorer = variant_scorers.CenterMaskScorer(
        requested_output=output_enum,
        width=mask_width,
        aggregation_type=variant_scorers.AggregationType.DIFF_MEAN,
    )
    
    # Score all ISM variants
    variant_scores = dna_model.score_ism_variants(
        interval=sequence_interval,
        ism_interval=ism_interval,
        variant_scorers=[variant_scorer],
    )
    
    # Extract scores for target cell line/tissue
    def extract_target_scores(adata):
        values = adata.X[:, adata.var['ontology_curie'] == target_cell_line]
        if values.size == 0:
            # If target not found, use first available track
            values = adata.X[:, 0:1]
        assert values.size >= 1
        return values.flatten()[0]
    
    # Create ISM matrix
    ism_result = ism.ism_matrix(
        [extract_target_scores(x[0]) for x in variant_scores],
        variants=[v[0].uns['variant'] for v in variant_scores],
    )
    
    artifacts = []
    
    # Save ISM matrix
    ism_matrix_file = OUTPUT_DIR / f"{out_prefix}_ism_matrix.csv"
    pd.DataFrame(ism_result).to_csv(ism_matrix_file, index=False)
    artifacts.append({
        "description": "ISM contribution matrix",
        "path": str(ism_matrix_file.resolve())
    })
    
    # Create sequence logo plot
    plt.figure(figsize=(35, 6))
    plot_components.plot(
        [
            plot_components.SeqLogo(
                scores=ism_result,
                scores_interval=ism_interval,
                ylabel=f'ISM {target_cell_line} {output_type}',
            )
        ],
        interval=ism_interval,
        fig_width=35,
    )
    
    logo_file = OUTPUT_DIR / f"{out_prefix}_sequence_logo.png"
    plt.savefig(logo_file, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts.append({
        "description": "ISM sequence logo plot",
        "path": str(logo_file.resolve())
    })
    
    return {
        "message": f"ISM analysis completed: {len(variant_scores)} variants scored ({ism_width} positions)",
        "reference": "alphagenome/colabs/quick_start.ipynb",
        "artifacts": artifacts
    }

@quick_start_mcp.tool
def mouse_predictions(
    api_key: Annotated[str, "AlphaGenome API key for authentication"],
    sequence: Annotated[str | None, "DNA sequence for sequence prediction"] = 'GATTACA',
    chromosome: Annotated[str | None, "Mouse chromosome for interval prediction"] = "chr1", 
    start_position: Annotated[int | None, "Start position for interval prediction"] = 3000000,
    end_position: Annotated[int | None, "End position for interval prediction"] = 3000001,
    prediction_type: Annotated[Literal["sequence", "interval"], "Type of prediction to perform"] = "sequence",
    sequence_length: Annotated[Literal["2KB", "16KB", "100KB", "500KB", "1MB"], "Model input sequence length"] = "2KB",
    output_types: Annotated[list[str], "List of output types to predict"] = ["DNASE"],
    ontology_terms: Annotated[list[str], "List of ontology terms for mouse tissues"] = ["UBERON:0002048"],
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Make predictions for mouse sequences and genomic intervals using MUS_MUSCULUS organism.
    Input is mouse DNA sequence or coordinates and output is prediction metadata tables.
    """
    # Set up output prefix
    if out_prefix is None:
        out_prefix = f"mouse_predictions_{timestamp}"
        
    # Create DNA model
    dna_model = dna_client.create(api_key)
    
    # Convert sequence length to client constant
    length_map = {
        "2KB": dna_client.SEQUENCE_LENGTH_2KB,
        "16KB": dna_client.SEQUENCE_LENGTH_16KB,
        "100KB": dna_client.SEQUENCE_LENGTH_100KB,
        "500KB": dna_client.SEQUENCE_LENGTH_500KB,
        "1MB": dna_client.SEQUENCE_LENGTH_1MB
    }
    target_length = length_map[sequence_length]
    
    # Convert output types to client enums
    output_enums = []
    for output_type in output_types:
        output_enums.append(getattr(dna_client.OutputType, output_type))
    
    artifacts = []
    
    if prediction_type == "sequence":
        # Sequence prediction for mouse
        if sequence is None:
            raise ValueError("sequence must be provided for sequence prediction")
            
        # Pad sequence to valid length
        padded_sequence = sequence.center(target_length, 'N')
        
        # Make mouse sequence prediction
        output = dna_model.predict_sequence(
            sequence=padded_sequence,
            organism=dna_client.Organism.MUS_MUSCULUS,
            requested_outputs=output_enums,
            ontology_terms=ontology_terms,
        )
        
        # Save results for each output type
        for output_type in output_types:
            track_data = getattr(output, output_type.lower())
            
            # Save values
            values_file = OUTPUT_DIR / f"{out_prefix}_{output_type.lower()}_values.csv"
            pd.DataFrame(track_data.values).to_csv(values_file, index=False)
            artifacts.append({
                "description": f"Mouse {output_type} prediction values",
                "path": str(values_file.resolve())
            })
            
            # Save metadata
            metadata_file = OUTPUT_DIR / f"{out_prefix}_{output_type.lower()}_metadata.csv"
            track_data.metadata.to_csv(metadata_file, index=False)
            artifacts.append({
                "description": f"Mouse {output_type} track metadata",
                "path": str(metadata_file.resolve())
            })
    
    elif prediction_type == "interval":
        # Interval prediction for mouse
        if chromosome is None or start_position is None or end_position is None:
            raise ValueError("chromosome, start_position, and end_position must be provided for interval prediction")
            
        # Create mouse interval
        interval = genome.Interval(chromosome, start_position, end_position).resize(target_length)
        
        # Make mouse interval prediction
        output = dna_model.predict_interval(
            interval=interval,
            organism=dna_client.Organism.MUS_MUSCULUS,
            requested_outputs=output_enums,
            ontology_terms=ontology_terms,
        )
        
        # Save results for each output type
        for output_type in output_types:
            track_data = getattr(output, output_type.lower())
            
            # Save metadata
            metadata_file = OUTPUT_DIR / f"{out_prefix}_{output_type.lower()}_metadata.csv"
            track_data.metadata.to_csv(metadata_file, index=False)
            artifacts.append({
                "description": f"Mouse {output_type} interval metadata",
                "path": str(metadata_file.resolve())
            })
    
    return {
        "message": f"Mouse {prediction_type} predictions completed for {len(output_types)} output types",
        "reference": "alphagenome/colabs/quick_start.ipynb",
        "artifacts": artifacts
    }