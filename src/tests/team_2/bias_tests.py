from src.utils.test_utils import *
import unittest


class BiasTest(unittest.TestCase):

    def setUp(self):
        self.model_path = "../../../model/gboost2_v0.onnx"  # replace with gboost2.onnx if you are working on the bad model
        self.modified_model_path = "../../../model/gboost2_v2.onnx"

    def test_gender_bias(self):
        # change this with your dataset's path
        data_path = '../../../data/Experiment_persoon_geslacht_vrouw/male_only.csv'

        # change this when you want to test a different feature
        feature = ['persoon_geslacht_vrouw']

        new_val = [1]  # 0 -> 1 for women, set this to the value you want to test for (e.g Age 40 -> Age 20)

        acc_original, acc_changed, p_value, dataset_size, original_checked_cnt, changed_checked_cnt = test_bias(
            data_path, self.model_path, feature, new_val)  # DO NOT CHANGE

        acc_original_2, acc_changed_2, p_value_2, dataset_size_2, original_checked_cnt_2, changed_checked_cnt_2 = test_bias(
            data_path, self.modified_model_path, feature, new_val)  # DO NOT CHANGE


        print('\nThe original model:\n')
        
        show_stats(dataset_size=dataset_size,
                    acc_original=acc_original,
                    f_name_original="Men",
                    acc_changed=acc_changed,
                    f_name_changed="Women",
                    original_checked_cnt=original_checked_cnt,
                    changed_checked_cnt=changed_checked_cnt,
                    p_val=p_value)

        print('\nAfter modifying the training data:\n')

        show_stats(dataset_size=dataset_size_2,
                    acc_original=acc_original_2,
                    f_name_original="Men",
                    acc_changed=acc_changed_2,
                    f_name_changed="Women",
                    original_checked_cnt=original_checked_cnt_2,
                    changed_checked_cnt=changed_checked_cnt_2,
                    p_val=p_value_2)

        self.assertGreater(p_value, 0.05, msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')
        self.assertGreater(p_value_2, 0.05, msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')


    def test_age_bias(self):
        # change this with your dataset's path
        data_path = 'data/Experiment_persoon_leeftijd_bij_onderzoek/20_only.csv'

        # change this when you want to test a different feature
        feature = ['persoon_leeftijd_bij_onderzoek']

        new_val = [50]  # 0 -> 1 for women, set this to the value you want to test for (e.g Age 40 -> Age 20)

        acc_original, acc_changed, p_value, dataset_size, original_checked_cnt, changed_checked_cnt = test_bias(
            data_path, self.model_path, feature, new_val)  # DO NOT CHANGE

        acc_original_2, acc_changed_2, p_value_2, dataset_size_2, original_checked_cnt_2, changed_checked_cnt_2 = test_bias(
            data_path, self.modified_model_path, feature, new_val)  # DO NOT CHANGE

        print('\nThe original model:\n')
        
        show_stats(dataset_size=dataset_size,
                    acc_original=acc_original,
                    f_name_original="20 Year Olds",
                    acc_changed=acc_changed,
                    f_name_changed="50 Year Olds",
                    original_checked_cnt=original_checked_cnt,
                    changed_checked_cnt=changed_checked_cnt,
                    p_val=p_value)

        print('\nAfter modifying the training data:\n')

        show_stats(dataset_size=dataset_size_2,
                    acc_original=acc_original_2,
                    f_name_original="20 Year Olds",
                    acc_changed=acc_changed_2,
                    f_name_changed="50 Year Olds",
                    original_checked_cnt=original_checked_cnt_2,
                    changed_checked_cnt=changed_checked_cnt_2,
                    p_val=p_value_2)

        self.assertGreater(p_value, 0.05, msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')
        self.assertGreater(p_value_2, 0.05,
                           msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')

    def test_history_of_development_bias(self):
        # change this with your dataset's path
        data_path = 'data/Experiment_pla_historie_ontwikkeling/0_only.csv'

        # change this when you want to test a different feature
        feature = ['pla_historie_ontwikkeling']

        new_val = [1]  # 0 -> 1 for women, set this to the value you want to test for (e.g Age 40 -> Age 20)

        acc_original, acc_changed, p_value, dataset_size, original_checked_cnt, changed_checked_cnt = test_bias(
            data_path, self.model_path, feature, new_val)  # DO NOT CHANGE

        acc_original_2, acc_changed_2, p_value_2, dataset_size_2, original_checked_cnt_2, changed_checked_cnt_2 = test_bias(
            data_path, self.modified_model_path, feature, new_val)  # DO NOT CHANGE

        print('\nThe original model:\n')
        print(
            f'Accuracy for sample of {dataset_size} people with no development action plan: {acc_original * 100:.1f}%')
        print(f'Accuracy for sample of {dataset_size} people with development action plan: {acc_changed * 100:.1f}%')
        print(
            f'Percentage checked amongst {dataset_size} people with no action plan: {original_checked_cnt * 100 / dataset_size:.1f}%')
        print(f'Percentage checked when everyone now has a plan: {changed_checked_cnt * 100 / dataset_size:.1f}%')

        print('\nAfter modifying the training data:\n')

        print(
            f'Accuracy for sample of {dataset_size_2} people with no development action plan: {acc_original_2 * 100:.1f}%')
        print(f'Accuracy for sample of {dataset_size_2} people with development action plan: {acc_changed_2 * 100:.1f}%')
        print(
            f'Percentage checked amongst {dataset_size_2} people with no action plan: {original_checked_cnt_2 * 100 / dataset_size_2:.1f}%')
        print(f'Percentage checked when everyone now has a plan: {changed_checked_cnt_2 * 100 / dataset_size_2:.1f}%')

        self.assertGreater(p_value, 0.05,
                           msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')
        self.assertGreater(p_value_2, 0.05,
                           msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')
        
        
    def test_case_single_mother(self):
        # change this with your dataset's path
        data_path = 'data/Experiment_single_mother_of_two/average_male.csv'

        # change this when you want to test a different feature
        feature = ['persoon_geslacht_vrouw', 'relatie_partner_huidige_partner___partner__gehuwd_',
                   'relatie_kind_huidige_aantal', 'belemmering_financiele_problemen',
                   'competentie_met_druk_en_tegenslag_omgaan', 'persoonlijke_eigenschappen_flexibiliteit_opm',
                   'persoonlijke_eigenschappen_uiterlijke_verzorging_opm']

        new_val = [1, 0, 2, 1, 1, 1, 1]  # 0 -> 1 for women, set this to the value you want to test for (e.g Age 40 -> Age 20)

        acc_original, acc_changed, p_value, dataset_size, original_checked_cnt, changed_checked_cnt = test_bias(
            data_path, self.model_path, feature, new_val)  # DO NOT CHANGE

        acc_original_2, acc_changed_2, p_value_2, dataset_size_2, original_checked_cnt_2, changed_checked_cnt_2 = test_bias(
            data_path, self.modified_model_path, feature, new_val)  # DO NOT CHANGE

        print('\nThe original model:\n')
        show_stats(dataset_size=dataset_size,
                    acc_original=acc_original,
                    f_name_original="average male with no children",
                    acc_changed=acc_changed,
                    f_name_changed="single mother with two children",
                    original_checked_cnt=original_checked_cnt,
                    changed_checked_cnt=changed_checked_cnt,
                    p_val=p_value)

        print('\nAfter modifying the training data:\n')

        show_stats(dataset_size=dataset_size_2,
                    acc_original=acc_original_2,
                    f_name_original="average male with no children",
                    acc_changed=acc_changed_2,
                    f_name_changed="single mother with two children",
                    original_checked_cnt=original_checked_cnt_2,
                    changed_checked_cnt=changed_checked_cnt_2,
                    p_val=p_value_2)

        self.assertGreater(p_value, 0.05, msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')
        self.assertGreater(p_value_2, 0.05,
                           msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')

    def test_case_high_risk_individual(self):
        # change this with your dataset's path
        data_path = 'data/Experiment_high_risk_profile/low_risk_50yr_men.csv'

        print('\nThe original model:\n')
        feature = ['persoon_leeftijd_bij_onderzoek', 'persoon_geslacht_vrouw',
               'relatie_kind_huidige_aantal', 'persoonlijke_eigenschappen_taaleis_voldaan',
               'belemmering_financiele_problemen']

        new_val = [20, 1, 1, 0, 1]  # 0 -> 1 for women, set this to the value you want to test for (e.g Age 40 -> Age 20)

        acc_original, acc_changed, p_value, dataset_size, original_checked_cnt, changed_checked_cnt = test_bias(
            data_path, self.model_path, feature, new_val)  # DO NOT CHANGE

        acc_original_2, acc_changed_2, p_value_2, dataset_size_2, original_checked_cnt_2, changed_checked_cnt_2 = test_bias(
            data_path, self.modified_model_path, feature, new_val)  # DO NOT CHANGE

        print('\nThe original model:\n')
        show_stats(dataset_size=dataset_size,
                    acc_original=acc_original,
                    f_name_original="50 year old men who knows Dutch and have no financial difficulties",
                    acc_changed=acc_changed,
                    f_name_changed="20 year old mother of two who does not know Dutch and struggle to pay bills",
                    original_checked_cnt=original_checked_cnt,
                    changed_checked_cnt=changed_checked_cnt,
                    p_val=p_value)

        print('\nAfter modifying the training data:\n')

        show_stats(dataset_size=dataset_size_2,
                    acc_original=acc_original_2,
                    f_name_original="50 year old men who knows Dutch and have no financial difficulties",
                    acc_changed=acc_changed_2,
                    f_name_changed="20 year old mother of two who does not know Dutch and struggle to pay bills",
                    original_checked_cnt=original_checked_cnt_2,
                    changed_checked_cnt=changed_checked_cnt_2,
                    p_val=p_value_2)

        self.assertGreater(p_value, 0.05,
                           msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')
        self.assertGreater(p_value_2, 0.05,
                           msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')
        
    # def test_case_immigrant_worker(self):
    #     # change this with your dataset's path
    #     data_path = 'data/Experiment_immigrant_with_roommates/average_male.csv'

    #     # change this when you want to test a different feature
    #     feature = ['persoonlijke_eigenschappen_spreektaal_anders', 'relatie_overig_historie_vorm__andere_inwonende',
    #                'persoonlijke_eigenschappen_taaleis_voldaan', 'adres_recentste_buurt_other', 'adres_recentste_wijk_delfshaven',
    #                'persoonlijke_eigenschappen_motivatie_opm', 'persoonlijke_eigenschappen_houding_opm']

    #     new_val = [1, 3, 0, 0, 1, 0, 0]  # 0 -> 1 for women, set this to the value you want to test for (e.g Age 40 -> Age 20)

    #     acc_original, acc_changed, p_value, dataset_size, original_checked_cnt, changed_checked_cnt = test_bias(
    #         data_path, self.model_path, feature, new_val)  # DO NOT CHANGE

    #     acc_original_2, acc_changed_2, p_value_2, dataset_size_2, original_checked_cnt_2, changed_checked_cnt_2 = test_bias(
    #         data_path, self.modified_model_path, feature, new_val)  # DO NOT CHANGE

    #     print('\nThe original model:\n')
    #     _show_stats(dataset_size=dataset_size,
    #                 acc_original=acc_original,
    #                 f_name_original="average dutch male",
    #                 acc_changed=acc_changed,
    #                 f_name_changed="immigrant worker with roommates",
    #                 original_checked_cnt=original_checked_cnt,
    #                 changed_checked_cnt=changed_checked_cnt,
    #                 p_val=p_value)

    #     print('\nAfter modifying the training data:\n')

    #     _show_stats(dataset_size=dataset_size_2,
    #                 acc_original=acc_original_2,
    #                 f_name_original="average dutch male",
    #                 acc_changed=acc_changed_2,
    #                 f_name_changed="immigrant worker with roommates",
    #                 original_checked_cnt=original_checked_cnt_2,
    #                 changed_checked_cnt=changed_checked_cnt_2,
    #                 p_val=p_value_2)

    #     self.assertGreater(p_value, 0.05, msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')
    #     self.assertGreater(p_value_2, 0.05,
    #                        msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')


    