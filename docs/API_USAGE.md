# API Usage

This document provides a guide to using Project Sidewalk's AI API. The API allows one to programmatically validate and analyze Project Sidewalk labels.

## Process a Label

This endpoint analyzes a single label at a specified coordinate within a given panorama. It returns a validation assessment and, for certain label types, a set of descriptive tags identifying potential issues (e.g., a curb ramp being too steep).

`POST /process`

#### Request

| Attribute    | Value                                               |
| :----------- | :-------------------------------------------------- |
| **URL**      | `https://sidewalk-ai-api.cs.washington.edu/process` |
| **Method**   | `POST`                                              |
| **Content-Type** | `multipart/form-data`                               |

#### Parameters

The following parameters must be sent in the body of the request.

| Parameter     | Data Type | Required | Description                                                                                                                                                                                                                                       |
| :------------ | :-------- | :------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `label_type`  | String    | Yes      | The type of accessibility feature being analyzed. Supported types with trained models include `curbramp`, `crosswalk`, `obstacle`, `surfaceproblem`, and `nocurbramp`.                                                                               |
| `panorama_id` | String    | Yes      | The unique identifier for the panorama image where the label is located.                                                                                                                                                                          |
| `x`           | Float     | Yes      | The horizontal coordinate of the label on the equirectangular panorama. This value **must be normalized** to be between 0 and 1, where `0.0` represents the far-left edge and `1.0` represents the far-right edge. To calculate, use: `pixel_x / image_width`. |
| `y`           | Float     | Yes      | The vertical coordinate of the label on the equirectangular panorama. This value **must be normalized** to be between 0 and 1, where `0.0` represents the top edge and `1.0` represents the bottom edge. To calculate, use: `pixel_y / image_height`. |

#### Response

##### Success Response (Code `200 OK`)

The API returns a JSON object containing the results of the analysis.

| Field                         | Data Type        | Description                                                                                                                                                                  |
| :---------------------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `label_type`                  | String           | The `label_type` that was processed, echoing the value provided in the request.                                                                                              |
| `tag_scores`                  | JSON Object      | An object where each key is a potential problem tag (e.g., `missing-tactile-warning`, `steep`) and its value is the model's confidence score for that tag being present.      |
| `tags`                        | Array of Strings | A list of problem tags that the model has confidently identified for the given label, based on exceeding a confidence threshold (currently 0.3).                                                 |
| `tagger_model_id`                        | String | The HuggingFace model id for the tagger model used to make the prediction.                                                 |
| `tagger_training_date`                        | String | A standard MM-DD-YYYY date indicating when the tagger model was trained.                                                 |
| `validation_result`           | String           | The model's overall conclusion about the validity of the label. A value of `"correct"` indicates that the model agrees with the presence of the specified label type. Likewise, `"incorrect"` signifies the opposite. |
| `validation_score`            | Float            | A confidence score (between 0 and 1) corresponding to the `validation_result`. A higher score indicates greater confidence in the validation conclusion.                         |
| `validation_estimated_accuracy` | Float            | An estimate of the model's accuracy for the given prediction. We calculate this because a model's confidence is not necessarily equal to the probability of it being correct. More info [here](https://towardsdatascience.com/a-comprehensive-guide-on-model-calibration-part-1-of-4-73466eb5e09a/).                                                                                                               |
| `validator_model_id`                        | String | The HuggingFace model id for the validator model used to make the prediction.                                                 |
| `validator_training_date`                        | String | A standard MM-DD-YYYY date indicating when the validator model was trained.                                                 |

#### Notes

The fields returned in the response are conditional on the `label_type` due to model availability.

-   **Tag-related fields** (`tag_scores`, `tags`, `tagger_model_id`, `tagger_training_date`) are returned for: `["crosswalk", "curbramp", "obstacle", "surfaceproblem"]`.
-   **Validation-related fields** (`validation_result`, `validation_score`, `validation_estimated_accuracy`, `validator_model_id`, `validator_training_date`) are returned for: `["crosswalk", "curbramp", "obstacle", "surfaceproblem", "nocurbramp"]`.

If a `label_type` is submitted that does not support a particular analysis, the corresponding fields will be omitted from the JSON response.
