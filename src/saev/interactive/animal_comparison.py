import marimo

__generated_with = "0.14.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import random

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import torch
    import tqdm
    return mo, os


@app.cell
def _(mo):
    mo.md(r"""Enter the directory of the the sorted taxon dataset""")
    return


@app.cell
def _():
    taxon_directory = '/local/scratch/wilson.5057/taxons'
    return


@app.cell
def _(mo, os):
    def make_dropdown(directory, rank):
        try:
            choices = sorted(
                os.listdir(directory)
            )
        except FileNotFoundError:
            choices = []
        return mo.ui.dropdown(choices, label=rank)
    return


app._unparsable_cell(
    r"""
    def make_genus_dictionary():
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    def get_photos():
    
                                    
                        
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
