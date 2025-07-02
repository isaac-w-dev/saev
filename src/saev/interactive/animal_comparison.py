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
    return (taxon_directory,)


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


    return (make_dropdown,)


@app.cell
def _():
    return


@app.cell
def _(kingdoms):
    print(kingdoms)
    return


@app.cell
def _(kingdoms, mo):
    mo.vstack([kingdoms], justify="start")

    return


@app.cell
def _(kingdoms, make_dropdown):
    Kingdom = kingdoms.value
    taxon_directory = taxon_directory +  '/' + Kingdom
    phylums = make_dropdown(taxon_directory, "Phylum: ")
    return phylums, taxon_directory


@app.cell
def _(mo, phylums):
    mo.vstack([phylums], justify="start")
    return


@app.cell
def _(make_dropdown, phylums, taxon_directory):
    Phylum = phylums.value
    classes = make_dropdown(taxon_directory + '/' + Phylum, "Class: ")
    return (classes,)


@app.cell
def _(classes, mo):
    mo.vstack([classes], justify="start")
    return


@app.cell
def _(classes, make_dropdown, taxon_directory):
    Class = classes.value
    orders = make_dropdown(taxon_directory, "Order: ")
    Order = orders.value
    families = make_dropdown(taxon_directory, "Family: ")
    family = families.value
    genuses = make_dropdown(taxon_directory, "Genus: ")
    Genus = genuses.value
    species = make_dropdown(taxon_directory, "Species: ")
    Species = species.value
    return


if __name__ == "__main__":
    app.run()
