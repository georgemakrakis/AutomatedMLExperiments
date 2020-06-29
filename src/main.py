from __future__ import print_function, unicode_literals
import regex
import os
import pathlib
import configparser

from pprint import pprint
from PyInquirer import style_from_dict, Token, prompt, Separator
from PyInquirer import Validator, ValidationError
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from experiment import run_experiment
# from src.experiment import run_experiment

style = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})


def handle_config():
    config_file = pathlib.Path('config.ini')
    if(config_file.exists() == False):
        pathlib.Path('config.ini').touch()
        return 1
    return 0


def handle_plots_folder():
    pots_directory = pathlib.Path('plots')
    if(pots_directory.exists() == False):
        pathlib.Path('plots').mkdir()
        return 1
    return 0


def get_dataset_files():
    file_list = os.listdir('dataset')
    return file_list


def handle_dataset_folder():
    dataset_directory = pathlib.Path('dataset')
    if(dataset_directory.exists() == False):
        pathlib.Path('dataset').mkdir()
        return False
    if(get_dataset_files() == []):
        return False
    return True


def read_config():

    parser = configparser.SafeConfigParser()
    parser.read('config.ini')

    experiment_config = dict()

    if(parser.has_section('MinMaxScaler')):
        try:
            int(parser['MinMaxScaler']['feature_range_low'])
            int(parser['MinMaxScaler']['feature_range_high'])
        except ValueError:
            return 1

        experiment_config['scaler'] = MinMaxScaler(copy=parser['MinMaxScaler']['copy'], feature_range=(
            int(parser['MinMaxScaler']['feature_range_low']), int(parser['MinMaxScaler']['feature_range_high'])))
    if(parser.has_section('StandardScaler')):
        experiment_config['scaler'] = StandardScaler()

    if(parser.has_section('KNeighborsClassifier')):
        try:
            int(parser['KNeighborsClassifier']['n_neighbors'])
        except ValueError:
            return 1

        experiment_config['classifier'] = KNeighborsClassifier(
            int(parser['KNeighborsClassifier']['n_neighbors']))

    if(parser.has_section('NaiveBayes')):
        experiment_config['classifier'] = GaussianNB()

    if(parser.has_section('split')):

        for name, value in parser.items('split'):
            if(value == 'train_test_split'):
                experiment_config['split'] = {
                    'split_method': 'train_test_split',
                    'test_size': float(
                        parser['split']['test_size'])
                }

            if(value == 'kfolds'):
                experiment_config['split'] = {
                    'split_method': 'kfolds',
                    'kfolds_folds_number': int(
                        parser['split']['kfolds_folds_number'])
                }

    if(parser.has_section('plots')):
        try:
            int(parser['plots']['plots_number'])
        except ValueError:
            return 1
        experiment_config['plots'] = {
            'plots_number': int(parser['plots']['plots_number'])
        }

    if(parser.has_section('dataset_files')):
        experiment_config['data'] = parser['dataset_files']['data']
        experiment_config['labels'] = parser['dataset_files']['labels']

    return experiment_config


def save_options(args):
    config = configparser.ConfigParser()

    if(args['scaler'] == 'MinMaxScaler'):
        config[args['scaler']] = {
            'feature_range_low': args['MinMaxScaler_feature_range_low'],
            'feature_range_high': args['MinMaxScaler_feature_range_high'],
            'copy': args['MinMaxScaler_copy']
        }
    if(args['scaler'] == 'StandardScaler'):
        config[args['scaler']] = {}

    if(args['classifier'] == 'KNeighborsClassifier'):
        config[args['classifier']] = {'algorithm': 'auto',
                                      'leaf_size': '30',
                                      'metric': 'euclidean',
                                      'metric_params': 'None',
                                      'n_jobs': 'None',
                                      'n_neighbors': args['KNeighborsClassifier_neighbors_number'],
                                      'p': '2',
                                      'weights': 'uniform'}

    if(args['classifier'] == 'NaiveBayes'):
        config[args['classifier']] = {}

    if(args['split_method'] == 'train_test_split'):
        config['split'] = {
            'split_method': args['split_method'],
            'test_size': args['train_test_split_test_size']
        }
    if(args['split_method'] == 'kfolds'):
        config['split'] = {
            'split_method': args['split_method'],
            'kfolds_folds_number': args['kfolds_folds_number']
        }
    if(len(args['dataset_files']) == 2):
        config['dataset_files'] = {
            'data': args['dataset_files'][0],
            'labels': args['dataset_files'][1]
        }

    config['plots'] = {'plots_number': args['plots_number']}

    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    return config


def primary_checks(args):

    # TODO Improve the way of checking stuff - maybe more parameters will be added
    if(os.stat("config.ini").st_size == 0 and ('classifier' not in args) and ('scaler' not in args) and ('split' not in args)):
        print('config file is empty and no parameters were supplied')
        return

    if(os.stat("config.ini").st_size != 0 and ('classifier' not in args) and ('scaler' not in args) and ('split' not in args)):
        print("no parameters where supplied - using saved parameters")

    if(('save_options' in args) and args['save_options'] == False):
        scaler, classifier, split, data, labels, plots = options_compose(
            args)

        run_experiment(scaler, classifier, split, data, labels, plots)
        return 0

    if(('save_options' in args) and args['save_options'] == True):
        print('saved and moved on')
        options = save_options(args)
        # print(options)

    config_values = read_config()

    if(config_values == {}):
        print('No supperted parameteres in config file')
        return
    if(config_values == 1):
        print('Wrong value(s) type to config file')
        return 1

    run_experiment(
        config_values['scaler'], config_values['classifier'], config_values['split'], config_values['data'],
        config_values['labels'], config_values['plots'])
    return 0


def options_compose(args):
    if(args['scaler'] == 'MinMaxScaler'):
        scaler = MinMaxScaler(copy=args['MinMaxScaler_copy'], feature_range=(
            args['MinMaxScaler_feature_range_low'], args['MinMaxScaler_feature_range_high']))
    if(args['scaler'] == 'StandardScaler'):
        scaler = StandardScaler()
    if(args['classifier'] == 'KNeighborsClassifier'):
        classifier = KNeighborsClassifier(
            args['KNeighborsClassifier_neighbors_number'])
    if(args['classifier'] == 'NaiveBayes'):
        classifier = GaussianNB()
    if(args['split_method'] == 'train_test_split'):
        split = {'split_method': 'train_test_split',
                 'test_size': args['train_test_split_test_size']}
    if(args['split_method'] == 'kfolds'):
        split = {'split_method': 'kfolds',
                 'kfolds_folds_number': args['kfolds_folds_number']}
    if(len(args['dataset_files']) == 2):
        data = args['dataset_files'][0]
        labels = args['dataset_files'][1]

    plots = {'plots_number': args['plots_number']}

    return scaler, classifier, split, data, labels, plots


class NumberValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a number',
                cursor_position=len(document.text))  # Move cursor to end


class NumberValidatorFloat(Validator):
    def validate(self, document):
        try:
            float(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a floating point number',
                cursor_position=len(document.text))  # Move cursor to end

# Remove function or use a main - init one


def parse_arguments():

    handle_config()

    handle_plots_folder()

    if(handle_dataset_folder() == False):
        print('No correct number of dataset files available')
        return

    dataset_files_list = []
    for file in get_dataset_files():
        dataset_files_list.append({'name': file})

    questions = [
        {
            'type': 'confirm',
            'name': 'run_options',
            'message': 'Do you want to load options from the config file and run the experiment?',
            'default': False
        },
        {
            'type': 'confirm',
            'name': 'save_options',
            'message': 'Do you want to save to config file?',
            'default': False,
            'when': lambda answers: answers['run_options'] == False
        },
        {
            'type': 'checkbox',
            'message': 'Select dataset files',
            'name': 'dataset_files',
            'choices': dataset_files_list,
            'when': lambda answers: answers['run_options'] == False
        },
        {
            'type': 'input',
            'name': 'plots_number',
            'message': 'How many plots do you want to produce',
            'validate': NumberValidator,
            'default': '4',
            'filter': lambda val: int(val),
            'when': lambda answers: answers['run_options'] == False
        },
        {
            'type': 'list',
            'name': 'scaler',
            'message': 'What type of scaler do you want to use?',
            'choices': ['MinMaxScaler', 'StandardScaler', 'None'],
            'default': 'None',
            'when': lambda answers: answers['run_options'] == False
        },
        {
            'type': 'input',
            'name': 'MinMaxScaler_feature_range_low',
            'message': 'Choose MinMaxScaler feature_range_low',
            'validate': NumberValidator,
            'default': '0',
            'filter': lambda val: int(val),
            'when': lambda answers: answers['run_options'] == False and answers['scaler'] == 'MinMaxScaler'
        },
        {
            'type': 'input',
            'name': 'MinMaxScaler_feature_range_high',
            'message': 'Choose MinMaxScaler feature_range_high',
            'validate': NumberValidator,
            'default': '1',
            'filter': lambda val: int(val),
            'when': lambda answers: answers['run_options'] == False and answers['scaler'] == 'MinMaxScaler'
        },
        {
            'type': 'confirm',
            'name': 'MinMaxScaler_copy',
            'message': 'Do you want to use copy for MinMaxScaler',
            'default': False,
            'when': lambda answers: answers['run_options'] == False and answers['scaler'] == 'MinMaxScaler'
        },
        {
            'type': 'list',
            'name': 'split_method',
            'message': 'What type of split do you want to do?',
            'choices': ['train_test_split', 'kfolds', 'None'],
            'default': 'None',
            'when': lambda answers: answers['run_options'] == False
        },
        {
            'type': 'input',
            'name': 'train_test_split_test_size',
            'message': 'Enter the test size (e.g.: 0.2 0.3, 0.5)',
            'default': '0.2',
            'filter': lambda val: float(val),
            'when': lambda answers: answers['run_options'] == False and answers['split_method'] == 'train_test_split'
        },
        {
            'type': 'input',
            'name': 'kfolds_folds_number',
            'message': 'Enter the number of folds',
            # Taken from the documentation (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
            'default': '5',
            'filter': lambda val: int(val),
            'when': lambda answers: answers['run_options'] == False and answers['split_method'] == 'kfolds'
        },
        {
            'type': 'list',
            'name': 'classifier',
            'message': 'What type of classifier do you want to use?',
            'choices': ['KNeighborsClassifier', 'NaiveBayes', 'None'],
            'default': 'None',
            'when': lambda answers: answers['run_options'] == False and answers['run_options'] == False
        },
        {
            'type': 'input',
            'name': 'KNeighborsClassifier_neighbors_number',
            'message': 'Enter number of neighboors',
            'default': '5',
            'validate': NumberValidator,
            'filter': lambda val: int(val),
            'when': lambda answers: answers['run_options'] == False and answers['classifier'] == 'KNeighborsClassifier'
        }
    ]

    answers = prompt(questions, style=style)

    # print('Arguments:')
    # pprint(answers)

    primary_checks(answers)


parse_arguments()
