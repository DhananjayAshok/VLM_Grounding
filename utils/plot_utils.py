import seaborn as sns
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error
import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self, parameters):
        if parameters is None:
            parameters = load_parameters()
        self.parameters = parameters
        self.size_params = {}
        sns.set_style("whitegrid")
        self.set_size_parameters()

    def set_size_parameters(self, scaler=1, font_size=16, labels_font_size=19, xtick_font_size=19, ytick_font_size=15, legend_font_size=16, title_font_size=20):
        plt.rcParams.update({'font.size': font_size * scaler})
        # set xlabel font size to 16
        plt.rcParams.update({'axes.labelsize': labels_font_size * scaler})
        # set x tick font size to 14
        plt.rcParams.update({'xtick.labelsize': xtick_font_size * scaler})
        # set y tick font size to 14
        plt.rcParams.update({'ytick.labelsize': ytick_font_size * scaler})
        # set title font size to 20
        plt.rcParams.update({'axes.titlesize': title_font_size * scaler})
        self.size_params["font_size"] = font_size * scaler
        self.size_params["labels_font_size"] = labels_font_size * scaler
        self.size_params["xtick_font_size"] = xtick_font_size * scaler
        self.size_params["ytick_font_size"] = ytick_font_size * scaler
        self.size_params["title_font_size"] = title_font_size * scaler
        self.size_params["legend_font_size"] = legend_font_size * scaler
        return

    def set_size_parameters_from_dict(self, size_params):
        """
        Set the size parameters from a dictionary. This trusts that the dictionary is correct and does not check for errors.
        """
        self.set_size_parameters(**size_params)
        return

    def get_size_input_number(self, key_name):
        while True:
            got = input(f"Enter the size for {key_name} (current value is {self.size_params[key_name]}): ")
            if got.strip() == "":
                return self.size_params[key_name]
            try:
                got = float(got)
                if got <= 0:
                    print(f"Got {got}, but it must be greater than 0")
                    continue
                return got
            except ValueError:
                print(f"Got {got}, but it must be a number")
                continue

    def test_sizes(self):
        if self.parameters["figure_force_save"]:
            self.parameters["logger"].warn("Parameters currently sets figure_force_save to True. This suggests you are running in an env without a display, but you cannot test sizes iteratively this way. This code may behave weirdly...")
        
        done = False
        while not done:
            print(f"Plot with sizes: ")
            print(f"{self.size_params}")
            plt.show()
            keepgoing = input("Do you want to keep trying different sizes? (only y will keep going):")
            if keepgoing.lower().strip() == "y":
                for key in self.size_params:
                    self.size_params[key] = self.get_size_input_number(key)
                self.set_size_parameters_from_dict(self.size_params)
            else:
                done = True
                break
        return

    def show(self, save_path=None, data_df=None):
        if self.parameters["figure_force_save"] and save_path is None:
            log_error(self.parameters["logger"], "Figure force save is enabled, but no save path was provided")
            return
        if self.parameters["figure_force_save"]:
            figure_path = self.parameters["figure_dir"] + f"/{save_path}"
            figure_dir = os.path.dirname(figure_path)
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)
            if data_df is not None:
                data_df.to_csv(f"{figure_path}.csv", index=False)
            plt.savefig(f"{figure_path}.pdf")
            plt.clf()
        else:
            plt.show()

    