"""
AlphaGenome visualization modality tour for different genomic data types.

This MCP Server provides 9 tools:
1. visualize_gene_expression: Visualize RNA_SEQ and CAGE gene expression predictions
2. visualize_variant_expression_effects: Show REF vs ALT variant effects on gene expression
3. visualize_custom_annotations: Plot custom annotations like polyadenylation sites
4. visualize_chromatin_accessibility: Visualize DNASE and ATAC chromatin accessibility
5. visualize_splicing_effects: Visualize splicing predictions with SPLICE_SITES, SPLICE_SITE_USAGE, SPLICE_JUNCTIONS
6. visualize_variant_splicing_effects: Show REF vs ALT variant effects on splicing with sashimi plots
7. visualize_histone_modifications: Visualize CHIP_HISTONE predictions with custom colors
8. visualize_tf_binding: Visualize CHIP_TF transcription factor binding predictions
9. visualize_contact_maps: Visualize CONTACT_MAPS DNA-DNA contact predictions

All tools extracted from `alphagenome/colabs/visualization_modality_tour.ipynb`.
"""

# Standard imports
from typing import Annotated, Literal, Any
import pandas as pd
import numpy as np
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime
import matplotlib.pyplot as plt

# AlphaGenome imports
from alphagenome import colab_utils
from alphagenome.data import gene_annotation, genome, track_data, transcript
from alphagenome.models import dna_client
from alphagenome.visualization import plot_components

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp_inputs" / "visualization_modality_tour"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp_outputs" / "visualization_modality_tour"

INPUT_DIR = Path(os.environ.get("VISUALIZATION_MODALITY_TOUR_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("VISUALIZATION_MODALITY_TOUR_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
visualization_modality_tour_mcp = FastMCP(name="visualization_modality_tour")

@visualization_modality_tour_mcp.tool
def visualize_gene_expression(
    chromosome: Annotated[str, "Chromosome name (e.g., 'chr22')"],
    start_position: Annotated[int, "Start genomic position"],
    end_position: Annotated[int, "End genomic position"],
    ontology_terms: Annotated[list, "List of ontology terms for tissues/cell types"] = ['UBERON:0001159', 'UBERON:0001155'],
    api_key: Annotated[str | None, "AlphaGenome API key"] = None,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Visualize RNA_SEQ and CAGE gene expression predictions for a genomic interval.
    Input is genomic coordinates and ontology terms and output is gene expression visualization plot.
    """
    if api_key is None:
        raise ValueError("API key must be provided")
    
    # Create interval and resize to supported length
    interval = genome.Interval(chromosome, start_position, end_position).resize(
        dna_client.SEQUENCE_LENGTH_1MB
    )
    
    # Create model client
    dna_model = dna_client.create(api_key)
    
    # Load gene annotations
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )
    longest_transcript_extractor = transcript.TranscriptExtractor(
        gene_annotation.filter_to_longest_transcript(gtf_transcript)
    )
    
    # Make predictions
    output = dna_model.predict_interval(
        interval=interval,
        requested_outputs={
            dna_client.OutputType.RNA_SEQ,
            dna_client.OutputType.CAGE,
        },
        ontology_terms=ontology_terms,
    )
    
    # Extract transcripts
    longest_transcripts = longest_transcript_extractor.extract(interval)
    
    # Build plot
    plot = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(longest_transcripts),
            plot_components.Tracks(
                tdata=output.rna_seq,
                ylabel_template='RNA_SEQ: {biosample_name} ({strand})\n{name}',
            ),
            plot_components.Tracks(
                tdata=output.cage,
                ylabel_template='CAGE: {biosample_name} ({strand})\n{name}',
            ),
        ],
        interval=interval,
        title='Predicted RNA Expression (RNA_SEQ, CAGE) for colon tissue',
    )
    
    # Save plot
    if out_prefix is None:
        out_prefix = f"gene_expression_{timestamp}"
    output_file = OUTPUT_DIR / f"{out_prefix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "message": "Gene expression visualization completed successfully",
        "reference": "alphagenome/colabs/visualization_modality_tour.ipynb",
        "artifacts": [
            {
                "description": "Gene expression visualization plot",
                "path": str(output_file.resolve())
            }
        ]
    }

@visualization_modality_tour_mcp.tool
def visualize_variant_expression_effects(
    chromosome: Annotated[str, "Chromosome name (e.g., 'chr22')"],
    position: Annotated[int, "Variant genomic position"],
    reference_bases: Annotated[str, "Reference allele sequence"],
    alternate_bases: Annotated[str, "Alternate allele sequence"],
    interval_start: Annotated[int, "Start position for prediction interval"],
    interval_end: Annotated[int, "End position for prediction interval"],
    gene_symbol: Annotated[str | None, "Gene symbol to zoom in on"] = None,
    ontology_terms: Annotated[list, "List of ontology terms for tissues/cell types"] = ['UBERON:0001159', 'UBERON:0001155'],
    api_key: Annotated[str | None, "AlphaGenome API key"] = None,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Visualize REF vs ALT variant effects on gene expression with overlaid tracks.
    Input is variant coordinates and interval and output is variant effect visualization plot.
    """
    if api_key is None:
        raise ValueError("API key must be provided")
    
    # Create variant and interval
    variant = genome.Variant(chromosome, position, reference_bases, alternate_bases)
    interval = genome.Interval(chromosome, interval_start, interval_end).resize(
        dna_client.SEQUENCE_LENGTH_1MB
    )
    
    # Create model client
    dna_model = dna_client.create(api_key)
    
    # Load gene annotations
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )
    longest_transcript_extractor = transcript.TranscriptExtractor(
        gene_annotation.filter_to_longest_transcript(gtf_transcript)
    )
    
    # Make variant predictions
    output = dna_model.predict_variant(
        interval=interval,
        variant=variant,
        requested_outputs={
            dna_client.OutputType.RNA_SEQ,
            dna_client.OutputType.CAGE,
        },
        ontology_terms=ontology_terms,
    )
    
    # Extract transcripts
    longest_transcripts = longest_transcript_extractor.extract(interval)
    
    # Determine plot interval
    if gene_symbol is not None:
        plot_interval = gene_annotation.get_gene_interval(gtf, gene_symbol=gene_symbol)
        plot_interval.resize_inplace(plot_interval.width + 1000)
    else:
        plot_interval = interval
    
    # Define colors for REF and ALT
    ref_alt_colors = {'REF': 'dimgrey', 'ALT': 'red'}
    
    # Build plot
    plot = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(longest_transcripts),
            plot_components.OverlaidTracks(
                tdata={
                    'REF': output.reference.rna_seq.filter_to_nonpositive_strand(),
                    'ALT': output.alternate.rna_seq.filter_to_nonpositive_strand(),
                },
                colors=ref_alt_colors,
                ylabel_template='{biosample_name} ({strand})\n{name}',
            ),
            plot_components.OverlaidTracks(
                tdata={
                    'REF': output.reference.cage.filter_to_nonpositive_strand(),
                    'ALT': output.alternate.cage.filter_to_nonpositive_strand(),
                },
                colors=ref_alt_colors,
                ylabel_template='{biosample_name} ({strand})\n{name}',
            ),
        ],
        annotations=[plot_components.VariantAnnotation([variant])],
        interval=plot_interval,
        title='Effect of variant on predicted RNA Expression in colon tissue',
    )
    
    # Save plot
    if out_prefix is None:
        out_prefix = f"variant_expression_effects_{timestamp}"
    output_file = OUTPUT_DIR / f"{out_prefix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "message": "Variant expression effects visualization completed successfully",
        "reference": "alphagenome/colabs/visualization_modality_tour.ipynb",
        "artifacts": [
            {
                "description": "Variant expression effects plot",
                "path": str(output_file.resolve())
            }
        ]
    }

@visualization_modality_tour_mcp.tool
def visualize_custom_annotations(
    chromosome: Annotated[str, "Chromosome name (e.g., 'chr22')"],
    start_position: Annotated[int, "Start genomic position"],
    end_position: Annotated[int, "End genomic position"],
    annotation_intervals: Annotated[list, "List of annotation intervals as [chr, start, end, strand, label] tuples"],
    plot_start: Annotated[int, "Start position for plotting window"],
    plot_end: Annotated[int, "End position for plotting window"],
    ontology_terms: Annotated[list, "List of ontology terms for tissues/cell types"] = ['UBERON:0001159', 'UBERON:0002048'],
    api_key: Annotated[str | None, "AlphaGenome API key"] = None,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Visualize RNA predictions with custom interval annotations like polyadenylation sites.
    Input is genomic coordinates and custom annotations and output is annotated RNA visualization plot.
    """
    if api_key is None:
        raise ValueError("API key must be provided")
    
    # Create interval
    interval = genome.Interval(chromosome, start_position, end_position).resize(
        dna_client.SEQUENCE_LENGTH_1MB
    )
    
    # Create model client
    dna_model = dna_client.create(api_key)
    
    # Load gene annotations
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )
    longest_transcript_extractor = transcript.TranscriptExtractor(
        gene_annotation.filter_to_longest_transcript(gtf_transcript)
    )
    
    # Make predictions
    output = dna_model.predict_interval(
        interval=interval,
        requested_outputs={
            dna_client.OutputType.RNA_SEQ,
        },
        ontology_terms=ontology_terms,
    )
    
    # Extract transcripts
    longest_transcripts = longest_transcript_extractor.extract(interval)
    
    # Create custom annotation intervals
    custom_intervals = []
    labels = []
    for ann in annotation_intervals:
        if len(ann) >= 5:
            chr_name, start, end, strand, label = ann[:5]
            custom_intervals.append(genome.Interval(chr_name, start, end, strand))
            labels.append(label)
    
    # Define plotting interval
    plot_interval = genome.Interval(chromosome, plot_start, plot_end, '-')
    
    # Build plot
    plot = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(longest_transcripts),
            plot_components.Tracks(
                tdata=output.rna_seq.filter_to_negative_strand(),
                ylabel_template='RNA_SEQ: {biosample_name} ({strand})\n{name}',
                shared_y_scale=True,
            )
        ],
        annotations=[
            plot_components.IntervalAnnotation(
                custom_intervals,
                alpha=1,
                labels=labels,
                label_angle=90
            )
        ],
        interval=plot_interval,
        title='Custom annotations with RNA expression',
    )
    
    # Save plot
    if out_prefix is None:
        out_prefix = f"custom_annotations_{timestamp}"
    output_file = OUTPUT_DIR / f"{out_prefix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "message": "Custom annotations visualization completed successfully",
        "reference": "alphagenome/colabs/visualization_modality_tour.ipynb",
        "artifacts": [
            {
                "description": "Custom annotations plot",
                "path": str(output_file.resolve())
            }
        ]
    }

@visualization_modality_tour_mcp.tool
def visualize_chromatin_accessibility(
    chromosome: Annotated[str, "Chromosome name (e.g., 'chr22')"],
    start_position: Annotated[int, "Start genomic position"],
    end_position: Annotated[int, "End genomic position"],
    variant_position: Annotated[int | None, "Variant position to highlight"] = None,
    variant_ref: Annotated[str | None, "Variant reference allele"] = None,
    variant_alt: Annotated[str | None, "Variant alternate allele"] = None,
    promoter_intervals: Annotated[list | None, "List of promoter intervals as [chr, start, end, name] tuples"] = None,
    window_size: Annotated[int, "Size of plotting window around variant"] = 8000,
    ontology_terms: Annotated[list, "List of ontology terms for tissues/cell types"] = ['UBERON:0000317', 'UBERON:0001155', 'UBERON:0001157', 'UBERON:0001159', 'UBERON:0004992', 'UBERON:0008971'],
    api_key: Annotated[str | None, "AlphaGenome API key"] = None,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Visualize DNASE and ATAC chromatin accessibility predictions for intestinal tissues.
    Input is genomic coordinates and optional variant information and output is chromatin accessibility plot.
    """
    if api_key is None:
        raise ValueError("API key must be provided")
    
    # Create interval
    interval = genome.Interval(chromosome, start_position, end_position).resize(
        dna_client.SEQUENCE_LENGTH_1MB
    )
    
    # Create model client
    dna_model = dna_client.create(api_key)
    
    # Load gene annotations
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )
    longest_transcript_extractor = transcript.TranscriptExtractor(
        gene_annotation.filter_to_longest_transcript(gtf_transcript)
    )
    
    # Make predictions
    output = dna_model.predict_interval(
        interval,
        requested_outputs={
            dna_client.OutputType.DNASE,
            dna_client.OutputType.ATAC,
        },
        ontology_terms=ontology_terms,
    )
    
    # Extract transcripts
    longest_transcripts = longest_transcript_extractor.extract(interval)
    
    # Prepare annotations
    annotations = []
    
    # Add variant annotation if provided
    if variant_position is not None and variant_ref is not None and variant_alt is not None:
        variant = genome.Variant(chromosome, variant_position, variant_ref, variant_alt)
        annotations.append(plot_components.VariantAnnotation([variant]))
        plot_interval = variant.reference_interval.resize(window_size)
    else:
        plot_interval = interval
    
    # Add promoter annotations if provided
    if promoter_intervals is not None:
        promoter_objs = []
        for prom in promoter_intervals:
            if len(prom) >= 4:
                chr_name, start, end, name = prom[:4]
                promoter_objs.append(genome.Interval(chr_name, start, end, name=name))
        if promoter_objs:
            annotations.append(plot_components.IntervalAnnotation(promoter_objs))
    
    # Build plot
    plot = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(longest_transcripts),
            plot_components.Tracks(
                tdata=output.dnase,
                ylabel_template='DNASE: {biosample_name} ({strand})\n{name}',
            ),
            plot_components.Tracks(
                tdata=output.atac,
                ylabel_template='ATAC: {biosample_name} ({strand})\n{name}',
            ),
        ],
        interval=plot_interval,
        annotations=annotations,
        title='Predicted chromatin accessibility (DNASE, ATAC) for colon tissue',
    )
    
    # Save plot
    if out_prefix is None:
        out_prefix = f"chromatin_accessibility_{timestamp}"
    output_file = OUTPUT_DIR / f"{out_prefix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "message": "Chromatin accessibility visualization completed successfully",
        "reference": "alphagenome/colabs/visualization_modality_tour.ipynb",
        "artifacts": [
            {
                "description": "Chromatin accessibility plot",
                "path": str(output_file.resolve())
            }
        ]
    }

@visualization_modality_tour_mcp.tool
def visualize_splicing_effects(
    chromosome: Annotated[str, "Chromosome name (e.g., 'chr22')"],
    start_position: Annotated[int, "Start genomic position"],
    end_position: Annotated[int, "End genomic position"],
    gene_symbol: Annotated[str | None, "Gene symbol to focus on"] = None,
    ontology_terms: Annotated[list, "List of ontology terms for tissues/cell types"] = ['UBERON:0001157', 'UBERON:0001159'],
    api_key: Annotated[str | None, "AlphaGenome API key"] = None,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Visualize splicing predictions including SPLICE_SITES, SPLICE_SITE_USAGE, and SPLICE_JUNCTIONS.
    Input is genomic coordinates and optional gene symbol and output is splicing effects plot.
    """
    if api_key is None:
        raise ValueError("API key must be provided")
    
    # Create interval
    interval = genome.Interval(chromosome, start_position, end_position).resize(
        dna_client.SEQUENCE_LENGTH_1MB
    )
    
    # Create model client
    dna_model = dna_client.create(api_key)
    
    # Load gene annotations
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )
    longest_transcript_extractor = transcript.TranscriptExtractor(
        gene_annotation.filter_to_longest_transcript(gtf_transcript)
    )
    
    # Make predictions
    output = dna_model.predict_interval(
        interval=interval,
        requested_outputs={
            dna_client.OutputType.RNA_SEQ,
            dna_client.OutputType.SPLICE_SITES,
            dna_client.OutputType.SPLICE_SITE_USAGE,
            dna_client.OutputType.SPLICE_JUNCTIONS,
        },
        ontology_terms=ontology_terms,
    )
    
    # Extract transcripts
    longest_transcripts = longest_transcript_extractor.extract(interval)
    
    # Determine plot interval
    if gene_symbol is not None:
        plot_interval = gene_annotation.get_gene_interval(gtf, gene_symbol=gene_symbol)
        plot_interval.resize_inplace(plot_interval.width + 1000)
    else:
        plot_interval = interval
    
    # Build plot
    plot = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(longest_transcripts),
            plot_components.Tracks(
                tdata=output.splice_sites.filter_to_negative_strand(),
                ylabel_template='SPLICE SITES: {name} ({strand})',
            ),
        ],
        interval=plot_interval,
        title='Predicted splicing effects for colon tissue',
    )
    
    # Save plot
    if out_prefix is None:
        out_prefix = f"splicing_effects_{timestamp}"
    output_file = OUTPUT_DIR / f"{out_prefix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "message": "Splicing effects visualization completed successfully",
        "reference": "alphagenome/colabs/visualization_modality_tour.ipynb",
        "artifacts": [
            {
                "description": "Splicing effects plot",
                "path": str(output_file.resolve())
            }
        ]
    }

@visualization_modality_tour_mcp.tool
def visualize_variant_splicing_effects(
    chromosome: Annotated[str, "Chromosome name (e.g., 'chr22')"],
    variant_position: Annotated[int, "Variant genomic position"],
    variant_ref: Annotated[str, "Variant reference allele"],
    variant_alt: Annotated[str, "Variant alternate allele"],
    interval_start: Annotated[int, "Start position for prediction interval"],
    interval_end: Annotated[int, "End position for prediction interval"],
    gene_symbol: Annotated[str | None, "Gene symbol to focus on"] = None,
    tissue_filter: Annotated[str, "Specific tissue to filter for sashimi plots"] = "Colon_Transverse",
    ontology_terms: Annotated[list, "List of ontology terms for tissues/cell types"] = ['UBERON:0001157', 'UBERON:0001159'],
    api_key: Annotated[str | None, "AlphaGenome API key"] = None,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Visualize REF vs ALT variant effects on splicing with sashimi plots and overlaid tracks.
    Input is variant coordinates and interval and output is variant splicing effects plot with sashimi arcs.
    """
    if api_key is None:
        raise ValueError("API key must be provided")
    
    # Create variant and interval
    variant = genome.Variant(chromosome, variant_position, variant_ref, variant_alt)
    interval = genome.Interval(chromosome, interval_start, interval_end).resize(
        dna_client.SEQUENCE_LENGTH_1MB
    )
    
    # Create model client
    dna_model = dna_client.create(api_key)
    
    # Load gene annotations
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )
    transcript_extractor = transcript.TranscriptExtractor(gtf_transcript)
    
    # Make variant predictions
    output = dna_model.predict_variant(
        interval=interval,
        variant=variant,
        requested_outputs={
            dna_client.OutputType.RNA_SEQ,
            dna_client.OutputType.SPLICE_SITES,
            dna_client.OutputType.SPLICE_SITE_USAGE,
            dna_client.OutputType.SPLICE_JUNCTIONS,
        },
        ontology_terms=ontology_terms,
    )
    
    # Extract all transcripts
    transcripts = transcript_extractor.extract(interval)
    
    # Determine plot interval
    if gene_symbol is not None:
        plot_interval = gene_annotation.get_gene_interval(gtf, gene_symbol=gene_symbol)
        plot_interval.resize_inplace(plot_interval.width + 1000)
    else:
        plot_interval = interval
    
    ref_output = output.reference
    alt_output = output.alternate
    
    # Define colors for REF and ALT
    ref_alt_colors = {'REF': 'dimgrey', 'ALT': 'red'}
    
    # Build plot
    plot = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(transcripts),
            plot_components.Sashimi(
                ref_output.splice_junctions
                .filter_to_strand('-')
                .filter_by_tissue(tissue_filter),
                ylabel_template='Reference {biosample_name} ({strand})\n{name}',
            ),
            plot_components.Sashimi(
                alt_output.splice_junctions
                .filter_to_strand('-')
                .filter_by_tissue(tissue_filter),
                ylabel_template='Alternate {biosample_name} ({strand})\n{name}',
            ),
            plot_components.OverlaidTracks(
                tdata={
                    'REF': ref_output.rna_seq.filter_to_nonpositive_strand(),
                    'ALT': alt_output.rna_seq.filter_to_nonpositive_strand(),
                },
                colors=ref_alt_colors,
                ylabel_template='RNA_SEQ: {biosample_name} ({strand})\n{name}',
            ),
            plot_components.OverlaidTracks(
                tdata={
                    'REF': ref_output.splice_sites.filter_to_nonpositive_strand(),
                    'ALT': alt_output.splice_sites.filter_to_nonpositive_strand(),
                },
                colors=ref_alt_colors,
                ylabel_template='SPLICE SITES: {name} ({strand})',
            ),
            plot_components.OverlaidTracks(
                tdata={
                    'REF': (
                        ref_output.splice_site_usage.filter_to_nonpositive_strand()
                    ),
                    'ALT': (
                        alt_output.splice_site_usage.filter_to_nonpositive_strand()
                    ),
                },
                colors=ref_alt_colors,
                ylabel_template=(
                    'SPLICE SITE USAGE: {biosample_name} ({strand})\n{name}'
                ),
            ),
        ],
        interval=plot_interval,
        annotations=[plot_components.VariantAnnotation([variant])],
        title='Predicted REF vs. ALT effects of variant in colon tissue',
    )
    
    # Save plot
    if out_prefix is None:
        out_prefix = f"variant_splicing_effects_{timestamp}"
    output_file = OUTPUT_DIR / f"{out_prefix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "message": "Variant splicing effects visualization completed successfully",
        "reference": "alphagenome/colabs/visualization_modality_tour.ipynb",
        "artifacts": [
            {
                "description": "Variant splicing effects plot",
                "path": str(output_file.resolve())
            }
        ]
    }

@visualization_modality_tour_mcp.tool
def visualize_histone_modifications(
    chromosome: Annotated[str, "Chromosome name (e.g., 'chr22')"],
    start_position: Annotated[int, "Start genomic position"],
    end_position: Annotated[int, "End genomic position"],
    include_tss_annotations: Annotated[bool, "Whether to include transcription start site annotations"] = True,
    ontology_terms: Annotated[list, "List of ontology terms for tissues/cell types"] = ['UBERON:0000317', 'UBERON:0001155', 'UBERON:0001157', 'UBERON:0001159'],
    api_key: Annotated[str | None, "AlphaGenome API key"] = None,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Visualize CHIP_HISTONE predictions with custom colors grouped by histone mark.
    Input is genomic coordinates and output is histone modifications plot with colored tracks.
    """
    if api_key is None:
        raise ValueError("API key must be provided")
    
    # Create interval
    interval = genome.Interval(chromosome, start_position, end_position).resize(
        dna_client.SEQUENCE_LENGTH_1MB
    )
    
    # Create model client
    dna_model = dna_client.create(api_key)
    
    # Load gene annotations
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )
    gtf_longest_transcript = gene_annotation.filter_to_longest_transcript(gtf_transcript)
    longest_transcript_extractor = transcript.TranscriptExtractor(gtf_longest_transcript)
    
    # Make predictions
    output = dna_model.predict_interval(
        interval=interval,
        requested_outputs={dna_client.OutputType.CHIP_HISTONE},
        ontology_terms=ontology_terms,
    )
    
    # Extract transcripts
    longest_transcripts = longest_transcript_extractor.extract(interval)
    
    # Reorder tracks by histone mark and apply colors
    reordered_chip_histone = output.chip_histone.select_tracks_by_index(
        output.chip_histone.metadata.sort_values('histone_mark').index
    )
    
    histone_to_color = {
        'H3K27AC': '#e41a1c',
        'H3K36ME3': '#ff7f00',
        'H3K4ME1': '#377eb8',
        'H3K4ME3': '#984ea3',
        'H3K9AC': '#4daf4a',
        'H3K27ME3': '#ffc0cb',
    }
    
    track_colors = (
        reordered_chip_histone.metadata['histone_mark']
        .map(lambda x: histone_to_color.get(x.upper(), '#000000'))
        .values
    )
    
    # Prepare annotations
    annotations = []
    if include_tss_annotations:
        # Extract TSS annotations
        gtf_tss = gene_annotation.extract_tss(gtf_longest_transcript)
        tss_as_intervals = [
            genome.Interval(
                chromosome=row.Chromosome,
                start=row.Start,
                end=row.End + 1000,  # Add extra 1Kb so the TSSs are visible
                name=row.gene_name,
            )
            for _, row in gtf_tss.iterrows()
        ]
        annotations.append(
            plot_components.IntervalAnnotation(
                tss_as_intervals, alpha=0.5, colors='blue'
            )
        )
    
    # Build plot
    plot = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(longest_transcripts),
            plot_components.Tracks(
                tdata=reordered_chip_histone,
                ylabel_template=(
                    'CHIP HISTONE: {biosample_name} ({strand})\n{histone_mark}'
                ),
                filled=True,
                track_colors=track_colors,
            ),
        ],
        interval=interval,
        annotations=annotations,
        despine_keep_bottom=True,
        title='Predicted histone modification markers in colon tissue',
    )
    
    # Save plot
    if out_prefix is None:
        out_prefix = f"histone_modifications_{timestamp}"
    output_file = OUTPUT_DIR / f"{out_prefix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "message": "Histone modifications visualization completed successfully",
        "reference": "alphagenome/colabs/visualization_modality_tour.ipynb",
        "artifacts": [
            {
                "description": "Histone modifications plot",
                "path": str(output_file.resolve())
            }
        ]
    }

@visualization_modality_tour_mcp.tool
def visualize_tf_binding(
    chromosome: Annotated[str, "Chromosome name (e.g., 'chr22')"],
    start_position: Annotated[int, "Start genomic position"],
    end_position: Annotated[int, "End genomic position"],
    transcription_factor: Annotated[str | None, "Specific transcription factor to visualize (e.g., 'CTCF')"] = None,
    min_max_prediction: Annotated[float, "Minimum maximum prediction value to include tracks"] = 8000.0,
    gene_symbol: Annotated[str | None, "Gene symbol to focus analysis on"] = None,
    include_tss_annotations: Annotated[bool, "Whether to include transcription start site annotations"] = True,
    ontology_terms: Annotated[list, "List of ontology terms for tissues/cell types"] = ['UBERON:0001159', 'UBERON:0001157', 'EFO:0002067', 'EFO:0001187'],
    api_key: Annotated[str | None, "AlphaGenome API key"] = None,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Visualize CHIP_TF transcription factor binding predictions with filtering and averaging options.
    Input is genomic coordinates and filtering parameters and output is TF binding visualization plot.
    """
    if api_key is None:
        raise ValueError("API key must be provided")
    
    # Create interval
    interval = genome.Interval(chromosome, start_position, end_position).resize(
        dna_client.SEQUENCE_LENGTH_1MB
    )
    
    # Create model client
    dna_model = dna_client.create(api_key)
    
    # Load gene annotations
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )
    gtf_longest_transcript = gene_annotation.filter_to_longest_transcript(gtf_transcript)
    longest_transcript_extractor = transcript.TranscriptExtractor(gtf_longest_transcript)
    transcript_extractor = transcript.TranscriptExtractor(gtf_transcript)
    
    # Make predictions
    output = dna_model.predict_interval(
        interval=interval,
        requested_outputs={dna_client.OutputType.CHIP_TF},
        ontology_terms=ontology_terms,
    )
    
    # Extract transcripts
    longest_transcripts = longest_transcript_extractor.extract(interval)
    all_transcripts = transcript_extractor.extract(interval)
    
    # Determine plot interval and filtering logic
    if gene_symbol is not None:
        gene_interval = gene_annotation.get_gene_interval(gtf, gene_symbol=gene_symbol)
        gene_interval.resize_inplace(gene_interval.width + 1000)
        
        # Filter based on gene interval
        max_predictions = output.chip_tf.slice_by_interval(
            gene_interval, match_resolution=True
        ).values.max(axis=0)
        
        # Get top 10 tracks for gene-specific analysis
        output_filtered = output.chip_tf.filter_tracks(
            (max_predictions >= np.sort(max_predictions)[-10])
        )
        plot_interval = gene_interval
        transcripts_to_use = all_transcripts
    else:
        # Filter by minimum max prediction globally
        output_filtered = output.chip_tf.filter_tracks(
            output.chip_tf.values.max(axis=0) > min_max_prediction
        )
        plot_interval = interval
        transcripts_to_use = longest_transcripts
    
    # Handle specific transcription factor averaging
    if transcription_factor is not None:
        # Filter to specific TF and create mean track
        tf_mask = output_filtered.metadata['transcription_factor'] == transcription_factor
        if tf_mask.any():
            mean_tf_values = output_filtered.values[:, tf_mask].mean(axis=1)
            
            # Create new TrackData object
            tdata_mean_tf = track_data.TrackData(
                values=mean_tf_values[:, None],
                metadata=pd.DataFrame({
                    'transcription_factor': [transcription_factor],
                    'name': ['mean'],
                    'strand': ['.']
                }),
                interval=output_filtered.interval,
                resolution=output_filtered.resolution,
            )
            
            track_data_to_plot = tdata_mean_tf
            ylabel_template = '{name} {transcription_factor}'
            plot_title = f'Predicted {transcription_factor} binding (mean across cell types)'
        else:
            raise ValueError(f"No tracks found for transcription factor: {transcription_factor}")
    else:
        track_data_to_plot = output_filtered
        ylabel_template = 'CHIP TF: {biosample_name} ({strand})\n{transcription_factor}'
        plot_title = 'Predicted TF-binding in K562, HepG2, and sigmoid colon.'
    
    # Prepare annotations
    annotations = []
    if include_tss_annotations:
        # Extract TSS annotations
        gtf_tss = gene_annotation.extract_tss(gtf_longest_transcript)
        tss_as_intervals = [
            genome.Interval(
                chromosome=row.Chromosome,
                start=row.Start,
                end=row.End + 1000,  # Add extra 1Kb so the TSSs are visible
                name=row.gene_name,
            )
            for _, row in gtf_tss.iterrows()
        ]
        annotations.append(
            plot_components.IntervalAnnotation(
                tss_as_intervals, alpha=0.3, colors='blue'
            )
        )
    
    # Build plot
    plot = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(transcripts_to_use),
            plot_components.Tracks(
                tdata=track_data_to_plot,
                ylabel_template=ylabel_template,
                filled=True,
            ),
        ],
        interval=plot_interval,
        annotations=annotations,
        despine_keep_bottom=True,
        title=plot_title,
    )
    
    # Save plot
    if out_prefix is None:
        tf_suffix = f"_{transcription_factor}" if transcription_factor else ""
        out_prefix = f"tf_binding{tf_suffix}_{timestamp}"
    output_file = OUTPUT_DIR / f"{out_prefix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "message": "TF binding visualization completed successfully",
        "reference": "alphagenome/colabs/visualization_modality_tour.ipynb",
        "artifacts": [
            {
                "description": "TF binding plot",
                "path": str(output_file.resolve())
            }
        ]
    }

@visualization_modality_tour_mcp.tool
def visualize_contact_maps(
    chromosome: Annotated[str, "Chromosome name (e.g., 'chr22')"],
    start_position: Annotated[int, "Start genomic position"],
    end_position: Annotated[int, "End genomic position"],
    colormap: Annotated[str, "Matplotlib colormap for contact map"] = 'autumn_r',
    vmax: Annotated[float, "Maximum value for colormap scaling"] = 1.0,
    ontology_terms: Annotated[list, "List of ontology terms for cell lines"] = ['EFO:0002824'],
    api_key: Annotated[str | None, "AlphaGenome API key"] = None,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Visualize CONTACT_MAPS DNA-DNA contact predictions showing topologically-associated domains.
    Input is genomic coordinates and colormap parameters and output is contact maps visualization plot.
    """
    if api_key is None:
        raise ValueError("API key must be provided")
    
    # Create interval
    interval = genome.Interval(chromosome, start_position, end_position).resize(
        dna_client.SEQUENCE_LENGTH_1MB
    )
    
    # Create model client
    dna_model = dna_client.create(api_key)
    
    # Load gene annotations
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )
    longest_transcript_extractor = transcript.TranscriptExtractor(
        gene_annotation.filter_to_longest_transcript(gtf_transcript)
    )
    
    # Make predictions
    output = dna_model.predict_interval(
        interval=interval,
        requested_outputs={dna_client.OutputType.CONTACT_MAPS},
        ontology_terms=ontology_terms,
    )
    
    # Extract transcripts
    longest_transcripts = longest_transcript_extractor.extract(interval)
    
    # Build plot
    plot = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(longest_transcripts),
            plot_components.ContactMaps(
                tdata=output.contact_maps,
                ylabel_template='{biosample_name}\n{name}',
                cmap=colormap,
                vmax=vmax,
            ),
        ],
        interval=interval,
        title='Predicted contact maps',
    )
    
    # Save plot
    if out_prefix is None:
        out_prefix = f"contact_maps_{timestamp}"
    output_file = OUTPUT_DIR / f"{out_prefix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "message": "Contact maps visualization completed successfully",
        "reference": "alphagenome/colabs/visualization_modality_tour.ipynb",
        "artifacts": [
            {
                "description": "Contact maps plot",
                "path": str(output_file.resolve())
            }
        ]
    }