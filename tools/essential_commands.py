"""
Essential commands for AlphaGenome API interaction covering data structures and operations.

This MCP Server provides 8 tools:
1. create_genomic_interval: Create genomic intervals for DNA regions
2. create_genomic_variant: Create genomic variants for genetic changes  
3. create_track_data: Create TrackData objects from arrays and metadata
4. create_variant_scores: Create AnnData objects for variant scoring results
5. genomic_interval_operations: Perform operations on genomic intervals
6. variant_interval_operations: Check variant overlaps with intervals  
7. track_data_operations: Filter, resize, slice and transform TrackData
8. track_data_resolution_conversion: Convert between track data resolutions

All tools extracted from `alphagenome/colabs/essential_commands.ipynb`.
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
from alphagenome.data import genome, track_data
from alphagenome.models import dna_client
import anndata

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp_inputs" / "essential_commands"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp_outputs" / "essential_commands"

INPUT_DIR = Path(os.environ.get("ESSENTIAL_COMMANDS_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("ESSENTIAL_COMMANDS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
essential_commands_mcp = FastMCP(name="essential_commands")

@essential_commands_mcp.tool
def create_genomic_interval(
    chromosome: Annotated[str, "Chromosome name (e.g., 'chr1', 'chr2')"] = "chr1",
    start: Annotated[int, "Start position (0-based)"] = 1000,
    end: Annotated[int, "End position (0-based, exclusive)"] = 1010,
    strand: Annotated[Literal["+", "-", "."], "DNA strand"] = ".",
    name: Annotated[str, "Interval name"] = "",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Create genomic intervals for DNA regions with basic properties and operations.
    Input is genomic coordinates and output is interval object with properties and operations results.
    """
    
    # Create genomic interval
    interval = genome.Interval(chromosome=chromosome, start=start, end=end, strand=strand, name=name)
    
    # Calculate basic properties
    results = {
        "interval": {
            "chromosome": interval.chromosome,
            "start": interval.start,
            "end": interval.end,
            "strand": interval.strand,
            "name": interval.name,
            "center": interval.center(),
            "width": interval.width,
        },
        "operations": {
            "resize_100bp": str(interval.resize(100)),
            "resize_1MB": str(interval.resize(dna_client.SEQUENCE_LENGTH_1MB)),
        }
    }
    
    # Save results
    prefix = out_prefix or f"genomic_interval_{timestamp}"
    output_file = OUTPUT_DIR / f"{prefix}.csv"
    
    # Flatten results for CSV
    rows = []
    for category, data in results.items():
        for key, value in data.items():
            rows.append({"category": category, "property": key, "value": str(value)})
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    
    return {
        "message": f"Genomic interval created: {interval.chromosome}:{interval.start}-{interval.end}",
        "reference": "alphagenome/colabs/essential_commands.ipynb",
        "artifacts": [
            {
                "description": "Genomic interval properties",
                "path": str(output_file.resolve())
            }
        ]
    }

@essential_commands_mcp.tool
def create_genomic_variant(
    chromosome: Annotated[str, "Chromosome name (e.g., 'chr3')"] = "chr3",
    position: Annotated[int, "1-based position"] = 10000,
    reference_bases: Annotated[str, "Reference sequence (e.g., 'A')"] = "A",
    alternate_bases: Annotated[str, "Alternate sequence (e.g., 'C')"] = "C",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Create genomic variants for genetic changes with reference interval operations.
    Input is variant coordinates and sequences and output is variant properties and intervals.
    """
    
    # Create genomic variant
    variant = genome.Variant(
        chromosome=chromosome,
        position=position,
        reference_bases=reference_bases,
        alternate_bases=alternate_bases
    )
    
    # Get reference interval and properties
    ref_interval = variant.reference_interval
    input_interval = ref_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)
    
    # Create test interval for overlap testing
    test_interval = genome.Interval(chromosome=chromosome, start=position+5, end=position+10)
    
    results = {
        "variant": {
            "chromosome": variant.chromosome,
            "position": variant.position,
            "reference_bases": variant.reference_bases,
            "alternate_bases": variant.alternate_bases,
            "reference_interval": str(ref_interval),
            "input_interval_width": input_interval.width,
        },
        "overlap_tests": {
            "reference_overlaps_test": variant.reference_overlaps(test_interval),
            "alternate_overlaps_test": variant.alternate_overlaps(test_interval),
        }
    }
    
    # Save results
    prefix = out_prefix or f"genomic_variant_{timestamp}"
    output_file = OUTPUT_DIR / f"{prefix}.csv"
    
    # Flatten results for CSV
    rows = []
    for category, data in results.items():
        for key, value in data.items():
            rows.append({"category": category, "property": key, "value": str(value)})
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    
    return {
        "message": f"Genomic variant created: {variant.chromosome}:{variant.position} {variant.reference_bases}>{variant.alternate_bases}",
        "reference": "alphagenome/colabs/essential_commands.ipynb",
        "artifacts": [
            {
                "description": "Genomic variant properties",
                "path": str(output_file.resolve())
            }
        ]
    }

@essential_commands_mcp.tool
def create_track_data(
    values_array: Annotated[str | None, "Path to CSV file with values array (shape: sequence_length x num_tracks)"] = None,
    track_names: Annotated[list[str], "List of track names"] = ["track1", "track1", "track2"],
    track_strands: Annotated[list[str], "List of track strands (+, -, .)"] = ["+", "-", "."],
    chromosome: Annotated[str, "Chromosome for interval"] = "chr1",
    start: Annotated[int, "Start position"] = 1000,
    end: Annotated[int, "End position"] = 1004,
    resolution: Annotated[int, "Base pair resolution"] = 1,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Create TrackData objects from user arrays and metadata with validation.
    Input is values array and track metadata and output is TrackData object properties.
    """
    
    if values_array is None:
        # Use tutorial example data if no input provided
        values = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]).astype(np.float32)
    else:
        # Load user data
        values = np.loadtxt(values_array, delimiter=',').astype(np.float32)
    
    # Create metadata
    metadata = pd.DataFrame({
        'name': track_names,
        'strand': track_strands,
    })
    
    # Create genomic interval 
    interval = genome.Interval(chromosome=chromosome, start=start, end=end)
    
    # Create TrackData object
    tdata = track_data.TrackData(
        values=values, 
        metadata=metadata, 
        resolution=resolution, 
        interval=interval
    )
    
    # Save results
    prefix = out_prefix or f"track_data_{timestamp}"
    values_file = OUTPUT_DIR / f"{prefix}_values.csv"
    metadata_file = OUTPUT_DIR / f"{prefix}_metadata.csv"
    
    # Save values and metadata
    np.savetxt(values_file, tdata.values, delimiter=',')
    tdata.metadata.to_csv(metadata_file, index=False)
    
    results = {
        "shape": tdata.values.shape,
        "resolution": tdata.resolution,
        "interval": str(tdata.interval) if tdata.interval else "None",
        "num_tracks": len(tdata.metadata),
        "track_names": tdata.metadata.name.tolist(),
        "track_strands": tdata.metadata.strand.tolist(),
    }
    
    return {
        "message": f"TrackData created with shape {tdata.values.shape} and {len(tdata.metadata)} tracks",
        "reference": "alphagenome/colabs/essential_commands.ipynb",
        "artifacts": [
            {
                "description": "Track data values",
                "path": str(values_file.resolve())
            },
            {
                "description": "Track metadata",
                "path": str(metadata_file.resolve())
            }
        ]
    }

@essential_commands_mcp.tool
def create_variant_scores(
    scores_array: Annotated[str | None, "Path to CSV file with scores array (shape: num_genes x num_tracks)"] = None,
    gene_ids: Annotated[list[str], "List of gene IDs"] = ["ENSG0001", "ENSG0002", "ENSG0003"],
    track_names: Annotated[list[str], "List of track names"] = ["track1", "track2"],
    track_strands: Annotated[list[str], "List of track strands"] = ["+", "-"],
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Create AnnData objects for variant scoring results with gene and track metadata.
    Input is scores array and metadata and output is AnnData object structure.
    """
    
    if scores_array is None:
        # Use tutorial example data if no input provided
        scores = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    else:
        # Load user data
        scores = np.loadtxt(scores_array, delimiter=',')
    
    # Create metadata
    gene_metadata = pd.DataFrame({'gene_id': gene_ids})
    track_metadata = pd.DataFrame({
        'name': track_names,
        'strand': track_strands
    })
    
    # Create AnnData object
    variant_scores = anndata.AnnData(
        X=scores, 
        obs=gene_metadata, 
        var=track_metadata
    )
    
    # Save results
    prefix = out_prefix or f"variant_scores_{timestamp}"
    scores_file = OUTPUT_DIR / f"{prefix}_scores.csv"
    gene_metadata_file = OUTPUT_DIR / f"{prefix}_gene_metadata.csv"
    track_metadata_file = OUTPUT_DIR / f"{prefix}_track_metadata.csv"
    
    # Save components
    np.savetxt(scores_file, variant_scores.X, delimiter=',')
    variant_scores.obs.to_csv(gene_metadata_file, index=False)
    variant_scores.var.to_csv(track_metadata_file, index=False)
    
    results = {
        "scores_shape": variant_scores.X.shape,
        "num_genes": variant_scores.n_obs,
        "num_tracks": variant_scores.n_vars,
        "gene_ids": variant_scores.obs.gene_id.tolist(),
        "track_names": variant_scores.var.name.tolist(),
    }
    
    return {
        "message": f"Variant scores AnnData created with {variant_scores.n_obs} genes and {variant_scores.n_vars} tracks",
        "reference": "alphagenome/colabs/essential_commands.ipynb",
        "artifacts": [
            {
                "description": "Variant scores matrix",
                "path": str(scores_file.resolve())
            },
            {
                "description": "Gene metadata",
                "path": str(gene_metadata_file.resolve())
            },
            {
                "description": "Track metadata",
                "path": str(track_metadata_file.resolve())
            }
        ]
    }

@essential_commands_mcp.tool
def genomic_interval_operations(
    interval1_chromosome: Annotated[str, "First interval chromosome"] = "chr1",
    interval1_start: Annotated[int, "First interval start"] = 1000,
    interval1_end: Annotated[int, "First interval end"] = 1010,
    interval2_chromosome: Annotated[str, "Second interval chromosome"] = "chr1", 
    interval2_start: Annotated[int, "Second interval start"] = 1005,
    interval2_end: Annotated[int, "Second interval end"] = 1015,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Perform operations on genomic intervals including overlaps, intersections and comparisons.
    Input is two genomic intervals and output is comparison operations results.
    """
    
    # Create intervals
    interval1 = genome.Interval(chromosome=interval1_chromosome, start=interval1_start, end=interval1_end)
    interval2 = genome.Interval(chromosome=interval2_chromosome, start=interval2_start, end=interval2_end)
    
    # Perform operations
    operations = {
        "interval1": str(interval1),
        "interval2": str(interval2),
        "interval1_center": interval1.center(),
        "interval1_width": interval1.width,
        "interval2_center": interval2.center(), 
        "interval2_width": interval2.width,
        "overlaps": interval1.overlaps(interval2),
        "contains": interval1.contains(interval2),
        "intersect": str(interval1.intersect(interval2)) if interval1.overlaps(interval2) else "No overlap",
        "interval1_resized_100": str(interval1.resize(100)),
        "interval2_resized_100": str(interval2.resize(100)),
    }
    
    # Save results
    prefix = out_prefix or f"genomic_interval_operations_{timestamp}"
    output_file = OUTPUT_DIR / f"{prefix}.csv"
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {"operation": key, "result": str(value)} 
        for key, value in operations.items()
    ])
    df.to_csv(output_file, index=False)
    
    return {
        "message": f"Genomic interval operations completed for {interval1} and {interval2}",
        "reference": "alphagenome/colabs/essential_commands.ipynb", 
        "artifacts": [
            {
                "description": "Interval operations results",
                "path": str(output_file.resolve())
            }
        ]
    }

@essential_commands_mcp.tool
def variant_interval_operations(
    variant_chromosome: Annotated[str, "Variant chromosome"] = "chr3",
    variant_position: Annotated[int, "Variant position (1-based)"] = 10000,
    variant_ref: Annotated[str, "Reference bases"] = "T",
    variant_alt: Annotated[str, "Alternate bases"] = "CGTCAAT",
    interval_chromosome: Annotated[str, "Test interval chromosome"] = "chr3",
    interval_start: Annotated[int, "Test interval start"] = 10005,
    interval_end: Annotated[int, "Test interval end"] = 10010,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Check variant overlaps with genomic intervals for reference and alternate alleles.
    Input is variant and interval coordinates and output is overlap test results.
    """
    
    # Create variant and interval
    variant = genome.Variant(
        chromosome=variant_chromosome,
        position=variant_position,
        reference_bases=variant_ref,
        alternate_bases=variant_alt,
    )
    
    interval = genome.Interval(
        chromosome=interval_chromosome, 
        start=interval_start, 
        end=interval_end
    )
    
    # Perform overlap tests
    results = {
        "variant": f"{variant.chromosome}:{variant.position} {variant.reference_bases}>{variant.alternate_bases}",
        "interval": str(interval),
        "reference_interval": str(variant.reference_interval),
        "reference_overlaps": variant.reference_overlaps(interval),
        "alternate_overlaps": variant.alternate_overlaps(interval),
        "variant_ref_length": len(variant.reference_bases),
        "variant_alt_length": len(variant.alternate_bases),
    }
    
    # Save results
    prefix = out_prefix or f"variant_interval_operations_{timestamp}"
    output_file = OUTPUT_DIR / f"{prefix}.csv"
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {"property": key, "value": str(value)} 
        for key, value in results.items()
    ])
    df.to_csv(output_file, index=False)
    
    return {
        "message": f"Variant interval operations completed: ref_overlaps={results['reference_overlaps']}, alt_overlaps={results['alternate_overlaps']}",
        "reference": "alphagenome/colabs/essential_commands.ipynb",
        "artifacts": [
            {
                "description": "Variant interval operations results", 
                "path": str(output_file.resolve())
            }
        ]
    }

@essential_commands_mcp.tool
def track_data_operations(
    values_array: Annotated[str | None, "Path to CSV file with values array"] = None,
    track_names: Annotated[list[str], "List of track names"] = ["track1", "track1", "track2"],
    track_strands: Annotated[list[str], "List of track strands"] = ["+", "-", "."],
    operation: Annotated[Literal["filter", "resize", "slice", "subset", "reverse_complement"], "Operation to perform"] = "filter",
    filter_strand: Annotated[Literal["+", "-", "."], "Strand to filter to"] = "+",
    resize_width: Annotated[int, "Width to resize to"] = 8,
    slice_start: Annotated[int, "Slice start position"] = 2,
    slice_end: Annotated[int, "Slice end position"] = 4,
    subset_names: Annotated[list[str], "Track names to subset to"] = ["track1"],
    chromosome: Annotated[str, "Chromosome"] = "chr1",
    start: Annotated[int, "Start position"] = 1000,
    end: Annotated[int, "End position"] = 1004,
    strand: Annotated[Literal["+", "-", "."], "Interval strand"] = "+",
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Filter, resize, slice and transform TrackData objects with strand-aware operations.
    Input is TrackData and operation parameters and output is transformed TrackData.
    """
    
    if values_array is None:
        # Use tutorial example data
        values = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]).astype(np.float32)
    else:
        values = np.loadtxt(values_array, delimiter=',').astype(np.float32)
    
    # Create metadata and interval
    metadata = pd.DataFrame({
        'name': track_names,
        'strand': track_strands,
    })
    
    interval = genome.Interval(chromosome=chromosome, start=start, end=end, strand=strand)
    tdata = track_data.TrackData(values=values, metadata=metadata, resolution=1, interval=interval)
    
    # Perform operation
    if operation == "filter":
        if filter_strand == "+":
            result_tdata = tdata.filter_to_positive_strand()
        elif filter_strand == "-":
            result_tdata = tdata.filter_to_negative_strand()
        else:
            result_tdata = tdata.filter_to_unstranded()
        operation_info = f"filtered to {filter_strand} strand"
    elif operation == "resize":
        result_tdata = tdata.resize(width=resize_width)
        operation_info = f"resized to width {resize_width}"
    elif operation == "slice":
        result_tdata = tdata.slice_by_positions(start=slice_start, end=slice_end)
        operation_info = f"sliced from {slice_start} to {slice_end}"
    elif operation == "subset":
        result_tdata = tdata.select_tracks_by_name(names=subset_names)
        operation_info = f"subset to tracks {subset_names}"
    elif operation == "reverse_complement":
        result_tdata = tdata.reverse_complement()
        operation_info = "reverse complemented"
    
    # Save results
    prefix = out_prefix or f"track_data_{operation}_{timestamp}"
    values_file = OUTPUT_DIR / f"{prefix}_values.csv"
    metadata_file = OUTPUT_DIR / f"{prefix}_metadata.csv"
    
    # Save values and metadata
    np.savetxt(values_file, result_tdata.values, delimiter=',')
    result_tdata.metadata.to_csv(metadata_file, index=False)
    
    return {
        "message": f"TrackData {operation_info}: shape {result_tdata.values.shape}",
        "reference": "alphagenome/colabs/essential_commands.ipynb",
        "artifacts": [
            {
                "description": f"Track data values after {operation}",
                "path": str(values_file.resolve())
            },
            {
                "description": f"Track metadata after {operation}",
                "path": str(metadata_file.resolve())
            }
        ]
    }

@essential_commands_mcp.tool  
def track_data_resolution_conversion(
    values_array: Annotated[str | None, "Path to CSV file with values array"] = None,
    track_names: Annotated[list[str], "List of track names"] = ["track1", "track1", "track2"],
    track_strands: Annotated[list[str], "List of track strands"] = ["+", "-", "."],
    original_resolution: Annotated[int, "Original resolution in bp"] = 1,
    target_resolution: Annotated[int, "Target resolution in bp"] = 2,
    chromosome: Annotated[str, "Chromosome"] = "chr1",
    start: Annotated[int, "Start position"] = 1000,
    end: Annotated[int, "End position"] = 1004,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Convert between different resolutions by upsampling or downsampling TrackData.
    Input is TrackData and resolution parameters and output is resolution-converted data.
    """
    
    if values_array is None:
        # Use tutorial example data
        values = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]).astype(np.float32)
    else:
        values = np.loadtxt(values_array, delimiter=',').astype(np.float32)
    
    # Create metadata and interval
    metadata = pd.DataFrame({
        'name': track_names,
        'strand': track_strands,
    })
    
    interval = genome.Interval(chromosome=chromosome, start=start, end=end)
    tdata = track_data.TrackData(
        values=values, 
        metadata=metadata, 
        resolution=original_resolution, 
        interval=interval
    )
    
    # Change resolution
    converted_tdata = tdata.change_resolution(resolution=target_resolution)
    
    # Save results
    prefix = out_prefix or f"track_data_resolution_{original_resolution}_to_{target_resolution}_{timestamp}"
    original_file = OUTPUT_DIR / f"{prefix}_original.csv"
    converted_file = OUTPUT_DIR / f"{prefix}_converted.csv"
    metadata_file = OUTPUT_DIR / f"{prefix}_metadata.csv"
    
    # Save original, converted values, and metadata
    np.savetxt(original_file, tdata.values, delimiter=',')
    np.savetxt(converted_file, converted_tdata.values, delimiter=',')
    converted_tdata.metadata.to_csv(metadata_file, index=False)
    
    return {
        "message": f"Resolution converted from {original_resolution}bp to {target_resolution}bp: {tdata.values.shape} -> {converted_tdata.values.shape}",
        "reference": "alphagenome/colabs/essential_commands.ipynb",
        "artifacts": [
            {
                "description": f"Original values at {original_resolution}bp resolution",
                "path": str(original_file.resolve())
            },
            {
                "description": f"Converted values at {target_resolution}bp resolution", 
                "path": str(converted_file.resolve())
            },
            {
                "description": "Track metadata",
                "path": str(metadata_file.resolve())
            }
        ]
    }