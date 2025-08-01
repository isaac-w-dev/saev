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
    import json
    return json, mo, os


@app.cell
def _(mo):
    def create_label(file_location):
        label = open(file_location).read().strip()
        label = label.split('_')
        label = ' '.join(label[1:])
        return mo.md(label)
    return (create_label,)


@app.cell
def _(mo):
    def display_image(image_location):
        return mo.image(src=image_location, width=200, rounded=True)
    return (display_image,)


@app.cell
def _(json):
    def display_metadata(metadata_location):
        with open(metadata_location) as metadata:
            data = json.load(metadata)
            return data
    return (display_metadata,)


@app.cell
def _(create_label, display_image, display_metadata, mo, os):
    def compile_info(species_instance_directory):
        info_block_array = []
        for folder in os.listdir(species_instance_directory):
            species_instance = species_instance_directory + '/' + folder
            info_block = ['Error', 'Error', 'Error', 'Error']
            for file in os.listdir(species_instance):
                file_location = species_instance + '/' + file
                if file[-4:] == '.png':
                    info_block[1] = display_image(file_location)
                if file[-4:] == '.txt':
                    info_block[0] = create_label(file_location)
                if file[-5:] == '.json':
                    checkpoint = mo.md(file[-22:-14])
                    info_block[2] = checkpoint
                    info_block[3] = display_metadata(file_location)
            info_dict = {'Name':info_block[0], 'Photo':info_block[1], 'Checkpoint':info_block[2], 'Neuron':info_block[3]['neuron'], 'Frequency' : 10 ** info_block[3]['log10_freq'], 'Value': 10 ** info_block[3]['log10_value']}
            info_block_array.append(info_dict)

        return mo.hstack(info_block_array, widths='equal', justify='start')
    return (compile_info,)


@app.cell
def _(os):
    def compile_species(directory):
        count = 0
        Genus_paths = []
        for Kingdom_directory in os.listdir(directory):
            Kingdom_path = directory + '/' + Kingdom_directory
            Kingdom = os.listdir(Kingdom_path)
            for Phylum_directory in Kingdom:
                Phylum_path = Kingdom_path + '/' + Phylum_directory
                Phylum = os.listdir(Phylum_path)
                for Class_directory in Phylum:
                    Class_path = Phylum_path + '/' + Class_directory
                    Class = os.listdir(Class_path)
                    for Order_directory in Class:
                        Order_path = Class_path + '/' + Order_directory
                        Order = os.listdir(Order_path)
                        for Family_directory in Order:
                            Family_path = Order_path + '/' + Family_directory
                            Family = os.listdir(Family_path)
                            for Genus_directory in Family:
                                Genus_path = Family_path + '/' + Genus_directory
                                Genus_paths.append(Genus_path)
                                count += 1
        return count - 1, Genus_paths

    return (compile_species,)


@app.cell
def _(compile_info, mo, os):
    def display_by_genus(genus_directory, get_i):
        genus_grid = []
        directory = genus_directory[get_i()]
        for species in os.listdir(directory):
            species_path = directory + '/' + species
            genus_grid.append(compile_info(species_path))
        return mo.vstack([genus_grid], heights='equal', justify='start')
    return (display_by_genus,)


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
def _(mo):
    mo.md(r"""Enter the directory of the the sorted taxon dataset""")
    return


@app.cell
def _():
    taxon_directory = '/local/scratch/wilson.5057/taxons'
    return (taxon_directory,)


@app.cell
def _(make_dropdown, taxon_directory):
    Kingdom = make_dropdown(taxon_directory, 'Kingdom')
    return (Kingdom,)


@app.cell
def _(Kingdom, mo):
    mo.hstack([Kingdom], justify="start")
    return


@app.cell
def _(Kingdom, make_dropdown, mo, taxon_directory):
    try:
        Phylum_directory = taxon_directory + '/' + Kingdom.value
        Phylum = make_dropdown(Phylum_directory, 'Phylum')
    except:
        print("Kingdom has not been set. Select Kingdom to continue")

    mo.hstack([Phylum], justify="start")

    return Phylum, Phylum_directory


@app.cell
def _(Phylum, Phylum_directory, make_dropdown, mo):
    try:
        Class_directory = Phylum_directory + '/' + Phylum.value
        Class = make_dropdown(Class_directory, 'Class')
    except:
        print("Phylum has not been set. Select Phylum to continue")

    mo.hstack([Class], justify="start")
    return Class, Class_directory


@app.cell
def _(Class, Class_directory, make_dropdown, mo):
    try:
        Order_directory = Class_directory + '/' + Class.value
        Order = make_dropdown(Order_directory, 'Order')
    except:
        print("Class has not been set. Select Class to continue")

    mo.hstack([Order], justify="start")
    return Order, Order_directory


@app.cell
def _(Order, Order_directory, make_dropdown, mo):
    try:
        Family_directory = Order_directory + '/' + Order.value
        Family = make_dropdown(Family_directory, 'Family')
    except:
        print("Order has not been set. Select Order to continue")

    mo.hstack([Family], justify="start")
    return Family, Family_directory


@app.cell
def _(Family, Family_directory, make_dropdown, mo):
    try:
        Genus_directory = Family_directory + '/' + Family.value
        Genus = make_dropdown(Genus_directory, 'Genus')
    except:
        print("Family has not been set. Select Family to continue")

    mo.hstack([Genus], justify="start")
    return Genus, Genus_directory


@app.cell
def _(Genus, Genus_directory, make_dropdown, mo):
    try:
        Species_directory = Genus_directory + '/' + Genus.value
        Species = make_dropdown(Species_directory, 'Species')
    except:
        print("Genus has not been set. Select Genus to continue")

    mo.hstack([Species], justify="start")
    return Species, Species_directory


@app.cell
def _(Species, Species_directory):
    try:
        Species_Instance_directory = Species_directory + '/' + Species.value
    except:
        print("Species has not been set. Select Species to continue")
    return (Species_Instance_directory,)


@app.cell
def _(Species_Instance_directory, compile_info, mo):
    species = compile_info(Species_Instance_directory)
    mo.vstack([species], heights='equal', justify='start')
    return


@app.cell
def _(mo):
    get_i, set_i = mo.state(0)
    return get_i, set_i


@app.cell
def _(compile_species, get_i, mo, set_i, taxon_directory):
    genus_count, genus_paths = compile_species(taxon_directory)
    genus_slider = mo.ui.slider(0, genus_count, value=get_i(), on_change=lambda i: set_i(i), full_width=True,)
    return genus_paths, genus_slider


@app.cell
def _(display_by_genus, genus_paths, get_i):
    grid = display_by_genus(genus_paths, get_i)
    return (grid,)


@app.cell
def _(mo):
    sort_by_freq_btn = mo.ui.run_button(label="Sort by frequency")
    sort_by_value_btn = mo.ui.run_button(label="Sort by value")
    sort_by_neuron_btn = mo.ui.run_button(label="Sort by neuron")
    return sort_by_freq_btn, sort_by_neuron_btn, sort_by_value_btn


@app.cell
def _(grid, sort_by_freq_btn, sort_by_neuron_btn, sort_by_value_btn):
    if sort_by_neuron_btn.value:
        new_grid = sorted(grid, key=lambda dct: int(dct["Neuron"]))
    elif sort_by_freq_btn.value:
        new_grid = sorted(grid, key=lambda dct: float(dct["Frequency"]))
    elif sort_by_value_btn.value:
        new_grid = sorted(grid, key=lambda dct: float(dct["Value"]))
    else:
        new_grid = grid
    return (new_grid,)


@app.cell
def _(mo, sort_by_freq_btn, sort_by_neuron_btn, sort_by_value_btn):
    mo.hstack([sort_by_freq_btn, sort_by_value_btn, sort_by_neuron_btn], justify="start")
    return


@app.cell
def _(genus_slider, mo, new_grid):
    mo.md(
        f"""
    {genus_slider}
    {new_grid}
    """
    )
    return


if __name__ == "__main__":
    app.run()
