import tkinter as tk
from medacy.pipelines import FDANanoDrugLabelPipeline
from medacy.ner import Model


def predict():
    global phrase, output, user_input, model
    phrase = user_input.get()

    # Generate the predictions
    anno = model.predict(phrase)
    anno_tuples = anno.get_entity_annotations()

    # Format them
    display_string = ""
    for type, _, _, entity in anno_tuples:
        this_string = "\"%s\" is a %s\n" % (entity, type)
        display_string += this_string

    phrase = display_string
    output.config(text=phrase)


def main():
    global phrase, output, user_input, model
    root = tk.Tk()
    root.title("MedaCy Demo")

    # Configure medaCy stuff
    pipeline = FDANanoDrugLabelPipeline(metamap=None, entities=['Drug'])  # TODO add all the entities we want in the demo
    model = Model(pipeline)

    # Configure the GUI
    tk.Label(root, text="Please enter a sentence: ").grid(row=0)
    user_input = tk.Entry(root)
    user_input.grid(row=0, column=1)
    phrase = ""
    tk.Button(root, text="Predict!", command=predict).grid(row=1, column=1, sticky=tk.W, pady=4)
    output = tk.Label(root, text=phrase)
    output.grid(row=2, column=1)

    root.mainloop()


if __name__ == "__main__":
    main()
