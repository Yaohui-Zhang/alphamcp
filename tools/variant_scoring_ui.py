"""
Interactive tutorial for scoring and visualizing single variants with comprehensive modality options.

This MCP Server provides 2 tools:
1. score_variant: Score a single variant with multiple variant scorers and save results
2. visualize_variant_effects: Generate comprehensive variant effect visualization across multiple modalities

All tools extracted from `alphagenome/colabs/variant_scoring_ui.ipynb`.
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
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp_inputs" / "variant_scoring_ui"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp_outputs" / "variant_scoring_ui"

INPUT_DIR = Path(os.environ.get("VARIANT_SCORING_UI_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("VARIANT_SCORING_UI_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
variant_scoring_ui_mcp = FastMCP(name="variant_scoring_ui")

# @variant_scoring_ui_mcp.tool
def score_variant(
    # Primary data inputs - variant specification
    variant_chromosome: Annotated[str, "Chromosome name (e.g., 'chr22')"] = "chr22",
    variant_position: Annotated[int, "Genomic position"] = 36201698,
    variant_reference_bases: Annotated[str, "Reference allele"] = "A",
    variant_alternate_bases: Annotated[str, "Alternate allele"] = "C",
    # Analysis parameters with tutorial defaults
    organism: Annotated[Literal["human", "mouse"], "Organism to analyze"] = "human",
    sequence_length: Annotated[Literal["2KB", "16KB", "100KB", "500KB", "1MB"], "Length of sequence around variant to predict"] = "1MB",
    api_key: Annotated[str | None, "API key for AlphaGenome model"] = None,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Score a single variant using multiple variant scorers with comprehensive analysis.
    Input is variant coordinates and parameters, output is variant scores table and downloadable CSV file.
    """
    from alphagenome import colab_utils
    from alphagenome.data import genome
    from alphagenome.models import dna_client, variant_scorers
    
    # Use provided API key or get from environment/colab
    if api_key:
        dna_model = dna_client.create(api_key)
    else:
        dna_model = dna_client.create(colab_utils.get_api_key())
    
    # Map organism string to enum
    organism_map = {
        'human': dna_client.Organism.HOMO_SAPIENS,
        'mouse': dna_client.Organism.MUS_MUSCULUS,
    }
    organism_enum = organism_map[organism]
    
    # Create variant object
    variant = genome.Variant(
        chromosome=variant_chromosome,
        position=variant_position,
        reference_bases=variant_reference_bases,
        alternate_bases=variant_alternate_bases,
    )
    
    # Get sequence length
    sequence_length_value = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f'SEQUENCE_LENGTH_{sequence_length}'
    ]
    
    # The input interval is derived from the variant (centered on it)
    interval = variant.reference_interval.resize(sequence_length_value)
    
    # Score variant
    variant_scores = dna_model.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=list(variant_scorers.RECOMMENDED_VARIANT_SCORERS.values()),
    )
    
    # Convert to tidy format
    df_scores = variant_scorers.tidy_scores(variant_scores)
    
    # Save results
    if out_prefix is None:
        out_prefix = f"variant_{variant_chromosome}_{variant_position}_{variant_reference_bases}_{variant_alternate_bases}"
    
    output_file = OUTPUT_DIR / f"{out_prefix}_scores_{timestamp}.csv"
    
    # Filter columns for display (remove internal columns)
    columns = [
        c for c in df_scores.columns if c not in ['variant_id', 'scored_interval']
    ]
    df_display = df_scores[columns]
    
    # Save full results
    df_scores.to_csv(output_file, index=False)
    
    return {
        "message": f"Variant scoring completed with {len(df_scores)} scores across modalities",
        "reference": "alphagenome/colabs/variant_scoring_ui.ipynb",
        "artifacts": [
            {
                "description": "Variant scores CSV",
                "path": str(output_file.resolve())
            }
        ]
    }

@variant_scoring_ui_mcp.tool
def visualize_variant_effects(
    # Primary data inputs - variant specification
    variant_chromosome: Annotated[str, "Chromosome name (e.g., 'chr22')"] = "chr22",
    variant_position: Annotated[int, "Genomic position"] = 36201698,
    variant_reference_bases: Annotated[str, "Reference allele"] = "A",
    variant_alternate_bases: Annotated[str, "Alternate allele"] = "C",
    # Analysis parameters with tutorial defaults
    organism: Annotated[Literal["human", "mouse"], "Organism to analyze"] = "human",
    sequence_length: Annotated[Literal["2KB", "16KB", "100KB", "500KB", "1MB"], "Length of sequence around variant to predict"] = "1MB",
    ontology_terms: Annotated[list[str], "List of cell and tissue ontology terms"] = None,
    # Gene annotation options
    plot_gene_annotation: Annotated[bool, "Include gene annotation in plot"] = True,
    plot_longest_transcript_only: Annotated[bool, "Show only longest transcript per gene"] = True,
    # Output types to plot
    plot_rna_seq: Annotated[bool, "Plot RNA-seq tracks"] = True,
    plot_cage: Annotated[bool, "Plot CAGE tracks"] = True,
    plot_atac: Annotated[bool, "Plot ATAC-seq tracks"] = False,
    plot_dnase: Annotated[bool, "Plot DNase tracks"] = False,
    plot_chip_histone: Annotated[bool, "Plot ChIP-seq histone tracks"] = False,
    plot_chip_tf: Annotated[bool, "Plot ChIP-seq transcription factor tracks"] = False,
    plot_splice_sites: Annotated[bool, "Plot splice sites"] = True,
    plot_splice_site_usage: Annotated[bool, "Plot splice site usage"] = False,
    plot_contact_maps: Annotated[bool, "Plot contact maps"] = False,
    plot_splice_junctions: Annotated[bool, "Plot splice junctions"] = False,
    # DNA strand filtering
    filter_to_positive_strand: Annotated[bool, "Filter tracks to positive strand only"] = False,
    filter_to_negative_strand: Annotated[bool, "Filter tracks to negative strand only"] = False,
    # Visualization options
    ref_color: Annotated[str, "Color for reference allele"] = "dimgrey",
    alt_color: Annotated[str, "Color for alternate allele"] = "red",
    plot_interval_width: Annotated[int, "Width of plot interval in base pairs"] = 43008,
    plot_interval_shift: Annotated[int, "Shift of plot interval from variant center"] = 0,
    api_key: Annotated[str | None, "API key for AlphaGenome model"] = None,
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Generate comprehensive variant effect visualization across multiple genomic modalities.
    Input is variant coordinates and visualization parameters, output is variant effect plots showing REF vs ALT predictions.
    """
    from alphagenome import colab_utils
    from alphagenome.data import gene_annotation, genome, transcript
    from alphagenome.models import dna_client
    from alphagenome.visualization import plot_components
    import matplotlib.pyplot as plt
    
    # Validate strand filtering parameters
    if filter_to_positive_strand and filter_to_negative_strand:
        raise ValueError(
            'Cannot specify both filter_to_positive_strand and '
            'filter_to_negative_strand.'
        )
    
    # Use provided API key or get from environment/colab
    if api_key:
        dna_model = dna_client.create(api_key)
    else:
        dna_model = dna_client.create(colab_utils.get_api_key())
    
    # Default ontology terms from tutorial
    if ontology_terms is None:
        ontology_terms = ['EFO:0001187', 'EFO:0002067', 'EFO:0002784']
    
    # Map organism string to enum
    organism_map = {
        'human': dna_client.Organism.HOMO_SAPIENS,
        'mouse': dna_client.Organism.MUS_MUSCULUS,
    }
    organism_enum = organism_map[organism]
    
    # Reference paths for gene annotation
    HG38_GTF_FEATHER = (
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    MM10_GTF_FEATHER = (
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'mm10/gencode.vM23.annotation.gtf.gz.feather'
    )
    
    # Create variant object
    variant = genome.Variant(
        chromosome=variant_chromosome,
        position=variant_position,
        reference_bases=variant_reference_bases,
        alternate_bases=variant_alternate_bases,
    )
    
    # Get sequence length
    sequence_length_value = dna_client.SUPPORTED_SEQUENCE_LENGTHS[
        f'SEQUENCE_LENGTH_{sequence_length}'
    ]
    
    # The input interval is derived from the variant (centered on it)
    interval = variant.reference_interval.resize(sequence_length_value)
    
    # Load gene annotation
    match organism_enum:
        case dna_client.Organism.HOMO_SAPIENS:
            gtf_path = HG38_GTF_FEATHER
        case dna_client.Organism.MUS_MUSCULUS:
            gtf_path = MM10_GTF_FEATHER
        case _:
            raise ValueError(f'Unsupported organism: {organism_enum}')

    import pandas as pd
    gtf = pd.read_feather(gtf_path)

    # Filter to protein-coding genes and highly supported transcripts
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )

    # Extractor for identifying transcripts in a region
    transcript_extractor = transcript.TranscriptExtractor(gtf_transcript)

    # Also define an extractor that fetches only the longest transcript per gene
    gtf_longest_transcript = gene_annotation.filter_to_longest_transcript(
        gtf_transcript
    )
    longest_transcript_extractor = transcript.TranscriptExtractor(
        gtf_longest_transcript
    )
    
    # Predict variant effects
    output = dna_model.predict_variant(
        interval=interval,
        variant=variant,
        organism=organism_enum,
        requested_outputs=[*dna_client.OutputType],
        ontology_terms=ontology_terms,
    )
    
    # Filter to DNA strand if requested
    ref, alt = output.reference, output.alternate

    if filter_to_positive_strand:
        ref = ref.filter_to_strand(strand='+')
        alt = alt.filter_to_strand(strand='+')
    elif filter_to_negative_strand:
        ref = ref.filter_to_strand(strand='-')
        alt = alt.filter_to_strand(strand='-')
    
    # Build plot components
    components = []
    ref_alt_colors = {'REF': ref_color, 'ALT': alt_color}

    # Gene and transcript annotation
    if plot_gene_annotation:
        if plot_longest_transcript_only:
            transcripts = longest_transcript_extractor.extract(interval)
        else:
            transcripts = transcript_extractor.extract(interval)
        components.append(plot_components.TranscriptAnnotation(transcripts))

    # Individual output type plots
    plot_map = {
        'plot_atac': (ref.atac, alt.atac, 'ATAC'),
        'plot_cage': (ref.cage, alt.cage, 'CAGE'),
        'plot_chip_histone': (ref.chip_histone, alt.chip_histone, 'CHIP_HISTONE'),
        'plot_chip_tf': (ref.chip_tf, alt.chip_tf, 'CHIP_TF'),
        'plot_contact_maps': (ref.contact_maps, alt.contact_maps, 'CONTACT_MAPS'),
        'plot_dnase': (ref.dnase, alt.dnase, 'DNASE'),
        'plot_rna_seq': (ref.rna_seq, alt.rna_seq, 'RNA_SEQ'),
        'plot_splice_junctions': (
            ref.splice_junctions,
            alt.splice_junctions,
            'SPLICE_JUNCTIONS',
        ),
        'plot_splice_sites': (ref.splice_sites, alt.splice_sites, 'SPLICE_SITES'),
        'plot_splice_site_usage': (
            ref.splice_site_usage,
            alt.splice_site_usage,
            'SPLICE_SITE_USAGE',
        ),
    }

    for key, (ref_data, alt_data, output_type) in plot_map.items():
        if eval(key) and ref_data is not None and ref_data.values.shape[-1] == 0:
            print(
                f'Requested plot for output {output_type} but no tracks exist in'
                ' output. This is likely because this output does not exist for your'
                ' ontologies or requested DNA strand.'
            )
        if eval(key) and ref_data and alt_data:
            match output_type:
                case 'CHIP_HISTONE':
                    ylabel_template = (
                        f'{output_type}: {{biosample_name}} ({{strand}})\n{{histone_mark}}'
                    )
                case 'CHIP_TF':
                    ylabel_template = (
                        f'{output_type}: {{biosample_name}}'
                        ' ({strand})\n{transcription_factor}'
                    )
                case 'CONTACT_MAPS':
                    ylabel_template = f'{output_type}: {{biosample_name}} ({{strand}})'
                case 'SPLICE_SITES':
                    ylabel_template = f'{output_type}: {{name}} ({{strand}})'
                case _:
                    ylabel_template = (
                        f'{output_type}: {{biosample_name}} ({{strand}})\n{{name}}'
                    )

            if output_type == 'CONTACT_MAPS':
                component = plot_components.ContactMapsDiff(
                    tdata=alt_data - ref_data,
                    ylabel_template=ylabel_template,
                )
                components.append(component)
            elif output_type == 'SPLICE_JUNCTIONS':
                ref_plot = plot_components.Sashimi(
                    ref_data,
                    ylabel_template='REF: ' + ylabel_template,
                )
                alt_plot = plot_components.Sashimi(
                    alt_data,
                    ylabel_template='ALT: ' + ylabel_template,
                )
                components.extend([ref_plot, alt_plot])
            else:
                component = plot_components.OverlaidTracks(
                    tdata={'REF': ref_data, 'ALT': alt_data},
                    colors=ref_alt_colors,
                    ylabel_template=ylabel_template,
                )
                components.append(component)

    # Validate plot interval width
    if plot_interval_width > interval.width:
        raise ValueError(
            f'plot_interval_width ({plot_interval_width}) must be less than '
            f'interval.width ({interval.width}).'
        )

    # Generate plot
    plot = plot_components.plot(
        components=components,
        interval=interval.shift(plot_interval_shift).resize(plot_interval_width),
        annotations=[
            plot_components.VariantAnnotation([variant]),
        ],
    )
    
    # Save plot
    if out_prefix is None:
        out_prefix = f"variant_{variant_chromosome}_{variant_position}_{variant_reference_bases}_{variant_alternate_bases}"
    
    output_file = OUTPUT_DIR / f"{out_prefix}_effects_{timestamp}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        "message": f"Variant visualization completed with {len(components)} plot components",
        "reference": "alphagenome/colabs/variant_scoring_ui.ipynb",
        "artifacts": [
            {
                "description": "Variant effects plot",
                "path": str(output_file.resolve())
            }
        ]
    }