# API Usage

This document provides a guide to using Project Sidewalk's AI API. We use this internally for AI auto validations and tagging.

## Process a Label

This endpoint analyzes a single label at a specified coordinate within a given panorama. It returns a validation assessment and, for certain label types, a set of descriptive tags identifying potential issues (e.g., a curb ramp being too steep).

`POST /process`

#### Request

| Attribute    | Value                                               |
| :----------- | :-------------------------------------------------- |
| **URL**      | `https://sidewalk-ai-api.cs.washington.edu/process` |
| **Method**   | `POST`                                              |
| **Content-Type** | `multipart/form-data`                               |
| **Authorization** | `Bearer <api key>`                               |

#### Authentication

Every request must include the API key as `Authorization: Bearer <api key>`. A missing or wrong key gets a `401`. The key is available to Project Sidewalk staff only.

#### Parameters

The following parameters are sent in the body of the request.

| Parameter     | Data Type | Required | Description                                                                                                                                                                                                                                       |
| :------------ | :-------- | :------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `label_type`  | String    | Yes      | The type of accessibility feature being analyzed. Supported types with trained models include `curbramp`, `crosswalk`, `obstacle`, `surfaceproblem`, and `nocurbramp`.                                                                               |
| `panorama_id` | String    | Yes      | The unique identifier for the panorama image where the label is located.                                                                                                                                                                          |
| `x`           | Float     | Yes      | The horizontal coordinate of the label on the equirectangular panorama. This value **must be normalized** to be between 0 and 1, where `0.0` represents the far-left edge and `1.0` represents the far-right edge. To calculate, use: `pixel_x / image_width`. |
| `y`           | Float     | Yes      | The vertical coordinate of the label on the equirectangular panorama. This value **must be normalized** to be between 0 and 1, where `0.0` represents the top edge and `1.0` represents the bottom edge. To calculate, use: `pixel_y / image_height`. |
| `city`        | String    | No       | The Project Sidewalk city id (e.g. `chicago-il`). When given, the API first looks for the panorama in the lab's local panorama cache for that city, and only downloads it from Google if it isn't there. |

#### Response

##### Success Response (Code `200 OK`)

The API returns a JSON object containing the results of the analysis.

| Field                         | Data Type        | Description                                                                                                                                                                  |
| :---------------------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `label_type`                  | String           | The `label_type` that was processed, echoing the value provided in the request.                                                                                              |
| `image_source`                | String           | Where the panorama came from: `"cache"` (the local panorama cache) or `"download"` (Google Street View).                                                                    |
| `api_version`                 | String           | The version of this API that produced the response.                                                                                                                          |
| `tag_scores`                  | JSON Object      | An object where each key is a potential problem tag (e.g., `missing tactile warning`, `steep`) and its value is the model's confidence score for that tag being present. Includes every tag for the label type, even ones the API never recommends. |
| `tags`                        | Array of Strings | Tags the model is confident are present. Each tag has its own threshold (in `tag_thresholds.json`), and a tag is listed when its score is at or above it. Some tags are never recommended as present. |
| `tags_not_present`            | Array of Strings | Tags the model is confident are absent: the tag's score is at or below its own "absent" threshold.                                                                           |
| `tagger_model_id`                        | String | The name of the tagger checkpoint used to make the prediction: the file `{tagger_model_id}.pth` in the HuggingFace repo `projectsidewalk/sidewalk-tagger-ai-{label_type}`. |
| `tagger_training_date`                        | String | A standard MM-DD-YYYY date indicating when the tagger model was trained.                                                 |
| `tagger_revision`                        | String | The HuggingFace commit of the tagger model, which identifies its exact weights.                                                 |
| `validation_result`           | String           | The model's overall conclusion about the validity of the label. A value of `"correct"` indicates that the model agrees with the presence of the specified label type. Likewise, `"incorrect"` signifies the opposite. |
| `validation_score`            | Float            | A confidence score (between 0 and 1) corresponding to the `validation_result`. A higher score indicates greater confidence in the validation conclusion.                         |
| `validation_estimated_accuracy` | Float            | An estimate of the model's accuracy for the given prediction. We calculate this because a model's confidence is not necessarily equal to the probability of it being correct. More info [here](https://towardsdatascience.com/a-comprehensive-guide-on-model-calibration-part-1-of-4-73466eb5e09a/).                                                                                                               |
| `validator_model_id`                        | String | The HuggingFace model id for the validator model used to make the prediction.                                                 |
| `validator_training_date`                        | String | A standard MM-DD-YYYY date indicating when the validator model was trained.                                                 |
| `validator_revision`                        | String | The HuggingFace commit of the validator model, which identifies its exact weights.                                                 |

##### Timing header

Responses also include a `Server-Timing` header with how long each step took in milliseconds, e.g. `pano;desc=cache;dur=310, depth;dur=95, ..., total;dur=530`. It's for debugging speed (`curl -i` shows it).

#### Notes

The fields returned in the response are conditional on the `label_type` due to model availability.

-   **Tag-related fields** (`tag_scores`, `tags`, `tags_not_present`, `tagger_model_id`, `tagger_training_date`, `tagger_revision`) are returned for: `["crosswalk", "curbramp", "obstacle", "surfaceproblem"]`.
-   **Validation-related fields** (`validation_result`, `validation_score`, `validation_estimated_accuracy`, `validator_model_id`, `validator_training_date`, `validator_revision`) are returned for: `["crosswalk", "curbramp", "obstacle", "surfaceproblem", "nocurbramp"]`.

If a `label_type` is submitted that does not support a particular analysis, the corresponding fields will be omitted from the JSON response.

##### Error Responses

Errors return a JSON object with an `error` message.

| Code  | When                                                                                          |
| :---- | :-------------------------------------------------------------------------------------------- |
| `400` | A required parameter is missing, `x`/`y` is outside 0–1, or `label_type` isn't supported.     |
| `401` | The API key is missing or wrong.                                                               |
| `500` | Model inference failed.                                                                        |
| `502` | The panorama couldn't be loaded from the cache or downloaded from Google (e.g. it no longer exists). |
