import unittest
import data_generator

class TestDataGeneratorMethods(unittest.TestCase):

    def setUp(self):
        # Parameters
        params = {'dim': (1363200, 1),
                  'batch_size': 64,
                  'n_classes': 6,
                  'n_channels': 1,
                  'shuffle': True}

        # Directories
        ok_directory = 'C:/Users/Tony/Downloads/Dataset2/OK/'
        nok_directory = 'C:/Users/Tony/Downloads/Dataset2/NOK/'

        labels = data_generator.DataGenerator.build_label_list(ok_directory=ok_directory, nok_directory=nok_directory)

        partition = data_generator.DataGenerator.build_partition(validation_amount=0.3, labels=labels)

        # Generators
        self.training_generator = data_generator.DataGenerator(partition['train'], labels, **params)
        self.validation_generator = data_generator.DataGenerator(partition['validation'], labels, **params)
        r=1
    def testOnEpochEnd(self):
        self.training_generator.on_epoch_end()
        self.assertEqual('1', '1')

suite = unittest.TestLoader().loadTestsFromTestCase(TestDataGeneratorMethods)
unittest.TextTestRunner(verbosity=2).run(suite)

