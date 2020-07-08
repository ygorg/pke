from overrides import overrides

from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('kpextractor_predictor')
class KPExtractorPredictor(Predictor):
    """"Predictor wrapper for the AcademicPaperClassifier"""
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        return output_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        try:
            title = json_dict['title']
            abstract = json_dict['abstract']
            source_string = title + ' . ' + abstract
            metadata = {'id': json_dict['id']}
        except KeyError:
            source_string = json_dict['source_string']
            metadata = {'id': '0'}
        return self._dataset_reader.text_to_instance(source_string=source_string, metadata=metadata)
