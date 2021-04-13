from action_classifier.lib_classifier import ClassifierOnlineTest

class MultiPersonClassifier(object):

    def __init__(self, model_path, classes, buff_size):

        self.classifier_dict = {}  # human id -> classifier of this person

        # Define a function for creating classifier for new people.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(model_path, classes, buff_size, human_id)

    def classify(self, dict_id2skeleton):
        ''' Classify the action_classifier type of each skeleton in dict_id2skeleton '''

        # Clear people not in view
        old_ids = set(self.classifier_dict)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.classifier_dict[human]

        # Predict each person's action_classifier
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():

            if id not in self.classifier_dict:  # add this new person
                self.classifier_dict[id] = self._create_classifier(id)

            classifier = self.classifier_dict[id]
            id2label[id] = classifier.predict(skeleton)  # predict label

        return id2label