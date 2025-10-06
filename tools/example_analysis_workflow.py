"""
Advanced TAL1 locus analysis workflow for T-cell acute lymphoblastic leukemia variants.

This MCP Server provides 3 tools:
1. visualize_tal1_variant_positions: Visualize genomic positions of oncogenic TAL1 variants
2. predict_variant_functional_impact: Predict functional effects of specific TAL1 variants
3. compare_oncogenic_vs_background_variants: Compare predicted effects of disease variants vs background

All tools extracted from `alphagenome/colabs/example_analysis_workflow.ipynb`.
"""

# Standard imports
from typing import Annotated, Literal, Any
import pandas as pd
import numpy as np
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime
import io
import itertools

# AlphaGenome imports  
from alphagenome import colab_utils
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
import plotnine as gg

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp_inputs" / "example_analysis_workflow"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp_outputs" / "example_analysis_workflow"

INPUT_DIR = Path(os.environ.get("EXAMPLE_ANALYSIS_WORKFLOW_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("EXAMPLE_ANALYSIS_WORKFLOW_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
example_analysis_workflow_mcp = FastMCP(name="example_analysis_workflow")

# Internal utility functions (kept from tutorial)
def _oncogenic_tal1_variants() -> pd.DataFrame:
    """Returns a dataframe of oncogenic T-ALL variants that affect TAL1."""
    variant_data = """
ID	CHROM	POS	REF	ALT	output	Study ID	Study Variant ID
Jurkat	chr1	47239296	C	CCGTTTCCTAACC	1	Mansour_2014
MOLT-3	chr1	47239296	C	ACC	1	Mansour_2014
Patient_1	chr1	47239296	C	AACG	1	Mansour_2014
Patient_2	chr1	47239291	CTAACC	TTTACCGTCTGTTAACGGC	1	Mansour_2014
Patient_3-5	chr1	47239296	C	ACG	1	Mansour_2014
Patient_6	chr1	47239296	C	ACC	1	Mansour_2014
Patient_7	chr1	47239295	AC	TCAAACTGGTAACC	1	Mansour_2014
Patient_8	chr1	47239296	C	AACC	1	Mansour_2014
new 3' enhancer 1	chr1	47212072	T	TGGGTAAACCGTCTGTTCAGCG	1	Smith_2023	UPNT802
new 3' enhancer 2	chr1	47212074	G	GAACGTT	1	Smith_2023	UPNT613
intergenic SNV 1	chr1	47230639	C	T	1	Liu_2020	SJALL043861_D1
intergenic SNV 2	chr1	47230639	C	T	1	Liu_2020	SJALL018373_D1
SJALL040467_D1	chr1	47239296	C	AACC	1	Liu_2020	SJALL040467_D1
PATBGC	chr1	47239296	C	AACC	1	Liu_2017	PATBGC
PATBTX	chr1	47239296	C	ACGGATATAACC	1	Liu_2017	PATBTX
PARJAY	chr1	47239296	C	ACGGAATTTCTAACC	1	Liu_2017	PARJAY
PARSJG	chr1	47239296	C	AACC	1	Liu_2017	PARSJG
PASYAJ	chr1	47239296	C	AACC	1	Liu_2017	PASYAJ
PATRAB	chr1	47239293	TTA	CTAACGG	1	Liu_2017	PATRAB
PAUBXP	chr1	47239296	C	ACC	1	Liu_2017	PAUBXP
PATENL	chr1	47239296	C	AACC	1	Liu_2017	PATENL
PARNXJ	chr1	47239296	C	ACG	1	Liu_2017	PARNXJ
PASXSI	chr1	47239296	C	AACC	1	Liu_2017	PASXSI
PASNEH	chr1	47239296	C	ACC	1	Liu_2017	PASNEH
PAUAFN	chr1	47239296	C	AACC	1	Liu_2017	PAUAFN
PARASZ	chr1	47239296	C	ACC	1	Liu_2017	PARASZ
PARWNW	chr1	47239296	C	ACC	1	Liu_2017	PARWNW
PASFKA	chr1	47239293	TTA	ACCGTTAATCAA	1	Liu_2017	PASFKA
PATEIT	chr1	47239296	C	AC	1	Liu_2017	PATEIT
PASMHF	chr1	47239296	C	AC	1	Liu_2017	PASMHF
PARJNX	chr1	47239296	C	AC	1	Liu_2017	PARJNX
PASYWF	chr1	47239296	C	AC	1	Liu_2017	PASYWF
"""
    return pd.read_table(io.StringIO(variant_data), sep='\t')

def _generate_background_variants(variant: genome.Variant, max_number: int = 100) -> pd.DataFrame:
    """Generates a dataframe of background variants for a given variant."""
    nucleotides = np.array(list('ACGT'), dtype='<U1')
    
    def generate_unique_strings(n, max_number, random_seed=42):
        """Generates unique random strings of length n."""
        rng = np.random.default_rng(random_seed)
        
        if 4**n < max_number:
            raise ValueError(
                'Cannot generate that many unique strings for the given length.'
            )
        
        generated_strings = set()
        while len(generated_strings) < max_number:
            indices = rng.integers(0, 4, size=n)
            new_string = ''.join(nucleotides[indices])
            if new_string != variant.alternate_bases:
                generated_strings.add(new_string)
        return list(generated_strings)
    
    permutations = []
    if 4 ** len(variant.alternate_bases) < max_number:
        # Get all
        for p in itertools.product(nucleotides, repeat=len(variant.alternate_bases)):
            permutations.append(''.join(p))
    else:
        # Sample some
        permutations = generate_unique_strings(len(variant.alternate_bases), max_number)
    
    ism_candidates = pd.DataFrame({
        'ID': ['mut_' + str(variant.position) + '_' + x for x in permutations],
        'CHROM': variant.chromosome,
        'POS': variant.position,
        'REF': variant.reference_bases,
        'ALT': permutations,
        'output': 0.0,
        'original_variant': variant.name,
    })
    return ism_candidates

def _vcf_row_to_variant(vcf_row: pd.Series) -> genome.Variant:
    """Parse a row of a vcf df into a genome.Variant."""
    variant = genome.Variant(
        chromosome=str(vcf_row.CHROM),
        position=int(vcf_row.POS),
        reference_bases=vcf_row.REF,
        alternate_bases=vcf_row.ALT,
        name=vcf_row.ID,
    )
    return variant

def _inference_df(qtl_df: pd.DataFrame, input_sequence_length: int) -> pd.DataFrame:
    """Returns a pd.DataFrame with variants and intervals ready for inference."""
    df = []
    for _, row in qtl_df.iterrows():
        variant = _vcf_row_to_variant(row)
        
        interval = genome.Interval(
            chromosome=row['CHROM'], start=row['POS'], end=row['POS']
        ).resize(input_sequence_length)
        
        df.append({
            'interval': interval,
            'variant': variant,
            'output': row['output'],
            'variant_id': row['ID'],
            'POS': row['POS'],
            'REF': row['REF'],
            'ALT': row['ALT'],
            'CHROM': row['CHROM'],
        })
    return pd.DataFrame(df)

def _oncogenic_and_background_variants(input_sequence_length: int, number_of_background_variants: int = 20) -> pd.DataFrame:
    """Generates a dataframe of all variants for this evaluation."""
    oncogenic_variants = _oncogenic_tal1_variants()
    
    variants = []
    for vcf_row in oncogenic_variants.itertuples():
        variants.append(
            genome.Variant(
                chromosome=str(vcf_row.CHROM),
                position=int(vcf_row.POS),
                reference_bases=vcf_row.REF,
                alternate_bases=vcf_row.ALT,
                name=vcf_row.ID,
            )
        )
    
    background_variants = pd.concat([
        _generate_background_variants(variant, number_of_background_variants)
        for variant in variants
    ])
    all_variants = pd.concat([oncogenic_variants, background_variants])
    return _inference_df(all_variants, input_sequence_length=input_sequence_length)

def _coarse_grained_mute_groups(eval_df):
    """Group variants by position and alternate allele length."""
    grp = []
    for row in eval_df.itertuples():
        if row.POS >= 47239290:  # MUTE site.
            if row.ALT_len > 4:
                grp.append('MUTE' + '_other')
            else:
                grp.append('MUTE' + '_' + str(row.ALT_len))
        else:
            grp.append(str(row.POS) + '_' + str(row.ALT_len))
    
    grp = pd.Series(grp)
    return pd.Categorical(grp, categories=sorted(grp.unique()), ordered=True)

@example_analysis_workflow_mcp.tool
def visualize_tal1_variant_positions(
    api_key: Annotated[str, "AlphaGenome API key for model access"],
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Visualize genomic positions of oncogenic TAL1 variants in T-cell acute lymphoblastic leukemia.
    Input is AlphaGenome API key and output is plot showing variant positions relative to TAL1 gene.
    """
    
    # Set up output path
    if out_prefix is None:
        out_prefix = f"tal1_variant_positions_{timestamp}"
    
    output_file = OUTPUT_DIR / f"{out_prefix}.png"
    
    # Create DNA model client
    dna_model = dna_client.create(api_key)
    
    # Load gene annotations
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    
    # Filter to protein-coding genes and highly supported transcripts
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )
    
    # Define an extractor that fetches only the longest transcript per gene
    gtf_longest_transcript = gene_annotation.filter_to_longest_transcript(gtf_transcript)
    longest_transcript_extractor = transcript_utils.TranscriptExtractor(gtf_longest_transcript)
    
    # Define TAL1 interval
    tal1_interval = genome.Interval(
        chromosome='chr1', start=47209255, end=47242023, strand='-'
    )
    
    # Gather unique variant positions and plot labels
    oncogenic_variants = _oncogenic_tal1_variants()
    unique_positions = oncogenic_variants['POS'].unique()
    unique_positions.sort()
    
    # Manually define labels to avoid overplotting
    labels = [
        '47212072, 47212074',
        '',
        '47230639', 
        '47239291 - 47239296',
        '',
        '',
        '',
    ]
    
    # Build plot
    plot_obj = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(
                longest_transcript_extractor.extract(tal1_interval)
            ),
        ],
        annotations=[
            plot_components.VariantAnnotation(
                [
                    genome.Variant(
                        chromosome='chr1',
                        position=x,
                        reference_bases='N',
                        alternate_bases='N',
                    )
                    for x in unique_positions
                ],
                labels=labels,
                use_default_labels=False,
            )
        ],
        interval=tal1_interval,
        title='Positions of variants near TAL1',
    )
    
    # Save the plot
    plot_obj.savefig(str(output_file), dpi=300, bbox_inches='tight')
    
    return {
        "message": "TAL1 variant positions visualization completed successfully",
        "reference": "alphagenome/colabs/example_analysis_workflow.ipynb",
        "artifacts": [
            {
                "description": "TAL1 variant positions plot",
                "path": str(output_file.resolve())
            }
        ]
    }

@example_analysis_workflow_mcp.tool
def predict_variant_functional_impact(
    api_key: Annotated[str, "AlphaGenome API key for model access"],
    variant_index: Annotated[int, "Index of variant from oncogenic TAL1 variants list (0-based)"] = 0,
    ontology_terms: Annotated[list, "List of ontology terms for cell type context"] = ['CL:0001059'],
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Predict functional impact of specific TAL1 variant on gene expression, accessibility and histone marks.
    Input is API key and variant parameters and output is detailed functional impact plot and analysis.
    """
    
    # Set up output path
    if out_prefix is None:
        out_prefix = f"tal1_variant_impact_{timestamp}"
    
    output_plot = OUTPUT_DIR / f"{out_prefix}.png"
    
    # Create DNA model client
    dna_model = dna_client.create(api_key)
    
    # Load gene annotations
    gtf = pd.read_feather(
        'https://storage.googleapis.com/alphagenome/reference/gencode/'
        'hg38/gencode.v46.annotation.gtf.gz.feather'
    )
    
    # Filter to protein-coding genes and highly supported transcripts
    gtf_transcript = gene_annotation.filter_transcript_support_level(
        gene_annotation.filter_protein_coding(gtf), ['1']
    )
    
    # Define an extractor that fetches only the longest transcript per gene
    gtf_longest_transcript = gene_annotation.filter_to_longest_transcript(gtf_transcript)
    longest_transcript_extractor = transcript_utils.TranscriptExtractor(gtf_longest_transcript)
    
    # Define TAL1 interval
    tal1_interval = genome.Interval(
        chromosome='chr1', start=47209255, end=47242023, strand='-'
    )
    
    # Get the variant of interest
    oncogenic_variants = _oncogenic_tal1_variants()
    variant = _vcf_row_to_variant(oncogenic_variants.iloc[variant_index])
    
    # Make predictions for sequences containing the REF and ALT alleles
    output = dna_model.predict_variant(
        interval=tal1_interval.resize(2**20),
        variant=variant,
        requested_outputs={
            dna_client.OutputType.RNA_SEQ,
            dna_client.OutputType.CHIP_HISTONE,
            dna_client.OutputType.DNASE,
        },
        ontology_terms=ontology_terms,
    )
    
    # Build plot
    longest_transcripts = longest_transcript_extractor.extract(tal1_interval)
    plot_obj = plot_components.plot(
        [
            plot_components.TranscriptAnnotation(longest_transcripts),
            # RNA-seq tracks
            plot_components.Tracks(
                tdata=output.alternate.rna_seq.filter_to_nonpositive_strand()
                - output.reference.rna_seq.filter_to_nonpositive_strand(),
                ylabel_template='{biosample_name} ({strand})\n{name}',
                filled=True,
            ),
            # DNAse tracks  
            plot_components.Tracks(
                tdata=output.alternate.dnase.filter_to_nonpositive_strand()
                - output.reference.dnase.filter_to_nonpositive_strand(),
                ylabel_template='{biosample_name} ({strand})\n{name}',
                filled=True,
            ),
            # Chip histone
            plot_components.Tracks(
                tdata=output.alternate.chip_histone.filter_to_nonpositive_strand()
                - output.reference.chip_histone.filter_to_nonpositive_strand(),
                ylabel_template='{biosample_name} ({strand})\n{name}',
                filled=True,
            ),
        ],
        annotations=[plot_components.VariantAnnotation([variant])],
        interval=tal1_interval,
        title=(
            'Effect of variant on predicted RNA Expression, DNAse, and ChIP-Histone'
            f' in CD34 positive HSC.\n{variant=}'
        ),
    )
    
    # Save the plot
    plot_obj.savefig(str(output_plot), dpi=300, bbox_inches='tight')
    
    return {
        "message": "TAL1 variant functional impact prediction completed successfully", 
        "reference": "alphagenome/colabs/example_analysis_workflow.ipynb",
        "artifacts": [
            {
                "description": "Variant functional impact plot",
                "path": str(output_plot.resolve())
            }
        ]
    }

@example_analysis_workflow_mcp.tool
def compare_oncogenic_vs_background_variants(
    api_key: Annotated[str, "AlphaGenome API key for model access"],
    number_of_background_variants: Annotated[int, "Number of background variants to generate per oncogenic variant"] = 3,
    input_sequence_length: Annotated[int, "Input sequence length for model predictions"] = 2**20,
    max_workers: Annotated[int, "Maximum number of workers for parallel scoring"] = 2,
    ontology_terms: Annotated[list, "List of ontology terms for cell type context"] = ['CL:0001059'],
    out_prefix: Annotated[str | None, "Output file prefix"] = None,
) -> dict:
    """
    Compare predicted TAL1 expression effects between oncogenic and background variants.
    Input is API key and analysis parameters and output is comparison plots and variant scoring data.
    """
    
    # Set up output paths
    if out_prefix is None:
        out_prefix = f"tal1_variant_comparison_{timestamp}"
    
    output_plots = []
    output_data = OUTPUT_DIR / f"{out_prefix}_variant_scores.csv"
    
    # Create DNA model client
    dna_model = dna_client.create(api_key)
    
    # Preparing variant groups
    eval_df = _oncogenic_and_background_variants(
        input_sequence_length=input_sequence_length, 
        number_of_background_variants=number_of_background_variants
    )
    
    # Additional annotations and variant groups
    eval_df['ALT_len'] = eval_df['ALT'].str.len()
    eval_df['variant_group'] = (
        eval_df['POS'].astype(str) + '_' + eval_df['ALT_len'].astype(str)
    )
    eval_df['output'] = eval_df['output'].fillna(0) != 0
    eval_df['coarse_grained_variant_group'] = _coarse_grained_mute_groups(eval_df)
    
    # Score the variants
    scores = dna_model.score_variants(
        intervals=eval_df['interval'].to_list(),
        variants=eval_df['variant'].to_list(),
        variant_scorers=[variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ']],
        max_workers=max_workers,
    )
    
    # Find the index corresponding to the TAL1 gene
    gene_index = scores[0][0].obs.query('gene_name == "TAL1"').index[0]
    # Find the index for our cell type of interest
    cell_type_index = (
        scores[0][0].var.query('ontology_curie == "CL:0001059"').index[0]
    )
    
    def get_tal1_score_for_cd34_cells(score_data):
        """Extracts the TAL1 expression score in CD34+ cells from the model output."""
        return score_data[gene_index, cell_type_index].X[0, 0]
    
    eval_df['tal1_diff_in_cd34'] = [
        get_tal1_score_for_cd34_cells(x[0]) for x in scores
    ]
    
    # Prepare plotting data
    plot_df = eval_df.loc[eval_df.REF != eval_df.ALT]
    
    # Turn variant into a string for easier processing
    plot_df['variant'] = plot_df['variant'].astype(str)
    
    plot_df = plot_df.loc[
        :,
        [
            'variant',
            'output', 
            'tal1_diff_in_cd34',
            'coarse_grained_variant_group',
        ],
    ].drop_duplicates()
    
    facet_title_by_group = {
        '47212072_22': 'chr1:47212072\n21 bp ins.',
        '47212074_7': 'chr1:47212072\n21 bp ins.',
        '47230639_1': 'chr1:47230639\nSNV',
        'MUTE_2': 'chr1:47239296\n1 bp ins.',
        'MUTE_3': 'chr1:47239296\n2 bp ins.',
        'MUTE_4': 'chr1:47239296\n3 bp ins.',
        'MUTE_other': 'chr1:47239296\n7-18 bp ins.',
    }
    
    plt_dict = {}
    
    # Generate plots for each group
    for group in plot_df.coarse_grained_variant_group.unique():
        subplot_df = pd.concat(
            [plot_df.assign(plot_group='density'), plot_df.assign(plot_group='rain')]
        )
        subplot_df = subplot_df[subplot_df.coarse_grained_variant_group == group]
        subplot_df = subplot_df[
            ~((subplot_df.plot_group == 'density') & (subplot_df.output))
        ]
        
        col_width = np.ptp(subplot_df.tal1_diff_in_cd34) / 200
        subplot_df['col_width'] = subplot_df['output'].map(
            {True: 1.5 * col_width, False: 1.25 * col_width}
        )
        
        plt_ = (
            gg.ggplot(subplot_df)
            + gg.aes(x='tal1_diff_in_cd34')
            + gg.geom_col(
                gg.aes(
                    y=1,
                    width='col_width',
                    fill='output',
                    x='tal1_diff_in_cd34',
                    alpha='output',
                ),
                data=subplot_df[subplot_df['plot_group'] == 'rain'],
            )
            + gg.geom_density(
                gg.aes(
                    x='tal1_diff_in_cd34',
                    fill='output',
                ),
                data=subplot_df[subplot_df['plot_group'] == 'density'],
                color='white',
            )
            + gg.facet_wrap('~output + plot_group', nrow=1, scales='free_x')
            + gg.scale_alpha_manual({True: 1, False: 0.3})
            + gg.scale_fill_manual({True: '#FAA41A', False: 'gray'})
            + gg.labs(title=facet_title_by_group[group])
            + gg.theme_minimal()
            + gg.geom_vline(xintercept=0, linetype='dotted')
            + gg.theme(
                figure_size=(1.2, 3),
                legend_position='none',
                axis_text_x=gg.element_blank(),
                panel_grid_major_x=gg.element_blank(),
                panel_grid_minor_x=gg.element_blank(),
                strip_text=gg.element_blank(),
                axis_title_y=gg.element_blank(),
                axis_title_x=gg.element_blank(),
                plot_title=gg.element_text(size=9),
            )
            + gg.scale_y_reverse()
            + gg.coord_flip()
        )
        
        plt_dict[group] = plt_
        
        # Save individual plots
        plot_output = OUTPUT_DIR / f"{out_prefix}_{group}.png"
        plt_.save(str(plot_output), dpi=300)
        output_plots.append({
            "description": f"Comparison plot for {group}",
            "path": str(plot_output.resolve())
        })
    
    # Save variant scoring data
    eval_df.to_csv(output_data, index=False)
    
    artifacts = [
        {
            "description": "Variant scoring data",
            "path": str(output_data.resolve())
        }
    ] + output_plots
    
    return {
        "message": "TAL1 oncogenic vs background variant comparison completed successfully",
        "reference": "alphagenome/colabs/example_analysis_workflow.ipynb", 
        "artifacts": artifacts
    }