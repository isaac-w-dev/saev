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


@app.function
def extract_taxonomy(file_location):
    label = open(file_location).read().strip()
    label = label.split('_')
    label = ' '.join(label[1:])
    return label


@app.cell
def _(mo):
    def display_image(image_location):
        return mo.image(src=image_location, width=200, rounded=True)
    return


@app.cell
def _(json):
    def extract_metadata(metadata_location):
        with open(metadata_location) as metadata:
            data = json.load(metadata)
            return data
    return (extract_metadata,)


@app.cell
def _(extract_metadata, os):
    def compile_info(species_instance_directory):
        info_block_array = []
        for folder in os.listdir(species_instance_directory):
            species_instance = species_instance_directory + '/' + folder
            info_block = ['Error', 'Error', 'Error', 'Error', 'Error', 'Error']
            for file in os.listdir(species_instance):
                file_location = species_instance + '/' + file
                if file[-4:] == '.txt':
                    info_block[0] = extract_taxonomy(file_location)
                if file[-4:] == '.png':
                    info_block[1] = file_location
                if file[-5:] == '.json':
                    checkpoint = file[-22:-14]
                    info_block[2] = checkpoint
                    metadata =  extract_metadata(file_location)
                    info_block[3] = metadata['neuron']
                    info_block[4] = metadata['log10_freq']
                    info_block[5] = metadata['log10_value']
            # info_dict = {'Name':info_block[0], 'Photo':info_block[1], 'Checkpoint':info_block[2], 'Neuron':info_block[3]['neuron'], 'Frequency' : 10 ** info_block[3]
            #     ['log10_freq'], 'Value': 10 ** info_block[3]['log10_value']}
            info_block_array.append(info_block)
        return info_block_array
    return (compile_info,)


@app.cell
def _(mo):
    def adjust_content_display(grid):
        adjusted_grid = []
        for list in grid:
            new_list = []
            for element in list:
                item = []
                item.append(mo.md(f'Name: {element[0]}'))
                item.append(mo.image(src=element[1], width=200, rounded=True))
                item.append(mo.md(f'Checkpoint: {element[2]}'))
                item.append(mo.md(f'Neuron: {element[3]}'))
                item.append(mo.md(f'Log10 Frequency: {element[4]}'))
                item.append(mo.md(f'Log10 Value: {element[5]}'))
                new_list.append(item)
            adjusted_grid.append(mo.hstack(new_list))
        return mo.vstack(adjusted_grid)
    return (adjust_content_display,)


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
def _(compile_info, os):
    def display_by_genus(genus_directory, get_i):
        genus_grid = []
        directory = genus_directory[get_i()]
        for species in os.listdir(directory):
            species_path = directory + '/' + species
            genus_grid.append(compile_info(species_path))
        return genus_grid
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
    return


@app.cell
def _(mo):
    get_i, set_i = mo.state(0)
    return get_i, set_i


@app.cell
def _():
    taxon_directory = '/local/scratch/wilson.5057/taxons'
    return (taxon_directory,)


@app.cell
def _(compile_species, get_i, mo, set_i, taxon_directory):
    genus_count, genus_paths = compile_species(taxon_directory)
    genus_slider = mo.ui.slider(0, genus_count, value=get_i(), on_change=lambda i: set_i(int(i)), full_width=True,)
    return genus_count, genus_paths, genus_slider


@app.cell
def _(display_by_genus, genus_paths, get_i):
    grid = display_by_genus(genus_paths, get_i)
    print(grid)
    return (grid,)


@app.cell
def _(genus_count, mo, set_i):
    sort_by_neuron_btn = mo.ui.run_button(label="Sort by neuron")
    sort_by_freq_btn = mo.ui.run_button(label="Sort by Log10 Frequency")
    sort_by_value_btn = mo.ui.run_button(label="Sort by Log10 Value")
    prev_btn = mo.ui.run_button(label="Previous", 
                               on_change=lambda  _: set_i(lambda g: (g - 1) % genus_count))
    next_btn = mo.ui.run_button(label="Next",
                               on_change=lambda  _: set_i(lambda g: (g + 1) % genus_count))
    return (
        next_btn,
        prev_btn,
        sort_by_freq_btn,
        sort_by_neuron_btn,
        sort_by_value_btn,
    )


@app.cell
def _(grid, sort_by_freq_btn, sort_by_neuron_btn, sort_by_value_btn):
    new_grid = []
    if sort_by_neuron_btn.value:
        for x in grid:
            # Sorts items by the value of neuron from lowest to highest
            new_grid.append(sorted(x, key=lambda list: list[3]))
    elif sort_by_freq_btn.value:
        for x in grid:
            # Sorts items by the value of neuron from lowest to highest
            new_grid.append(sorted(x, key=lambda list: list[4]))
    elif sort_by_value_btn.value:
        for x in grid:
            # Sorts items by the value of neuron from lowest to highest
            new_grid.append(sorted(x, key=lambda list: list[5]))
    else:
        new_grid = grid
    return (new_grid,)


@app.cell
def _(adjust_content_display, new_grid):
    restored_grid = adjust_content_display(new_grid)
    return (restored_grid,)


@app.cell
def _(
    mo,
    next_btn,
    prev_btn,
    sort_by_freq_btn,
    sort_by_neuron_btn,
    sort_by_value_btn,
):
    mo.vstack([mo.hstack([sort_by_neuron_btn, sort_by_freq_btn, sort_by_value_btn], justify="start"), mo.hstack([prev_btn, next_btn], justify="start")])

    return


@app.cell
def _(genus_slider, mo, restored_grid):
    mo.md(
        f"""
    {genus_slider}
    {restored_grid}
    """
    )
    return


if __name__ == "__main__":
    app.run()
