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
| `camera_pitch` | Float    | No       | The panorama camera's pitch in degrees (the label's `camera_pitch`), between -30 and 30 (real rig tilt is a few degrees, so a larger value is treated as a bad record). Absent, empty or `NaN` means unknown, and then no tilt correction is made. See [Camera pose and tilt correction](#camera-pose-and-tilt-correction). |
| `camera_roll`  | Float    | No       | The panorama camera's roll in degrees, between -30 and 30. Absent, empty or `NaN` counts as 0 (the tilt is still applied from `camera_pitch`). Used only together with `camera_pitch`. |
| `tilt_beta`    | Float    | No       | The fraction of the camera's tilt to correct for, between 0 and 1.5. Default `0.95`, measured for the pose the website stores. |
| `coord_frame`  | String   | No       | `level` (default) or `rig`: which frame `x` and `y` are in. `level` is a label placed by a person in the Project Sidewalk viewer; `rig` means `x`/`y` already point at a pixel of the panorama image (e.g. a label from the AI labeler), and they are never corrected. |

#### Response

##### Success Response (Code `200 OK`)

The API returns a JSON object containing the results of the analysis.

| Field                         | Data Type        | Description                                                                                                                                                                  |
| :---------------------------- | :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `label_type`                  | String           | The `label_type` that was processed, echoing the value provided in the request.                                                                                              |
| `image_source`                | String           | Where the panorama came from: `"cache"` (the local panorama cache) or `"download"` (Google Street View).                                                                    |
| `tilt_applied`                | Boolean          | `true` when the view was rotated to correct for the camera's tilt: `coord_frame` is `level`, `camera_pitch` was given, and the rotation is not zero. |
| `tilt_beta`                   | Float or null    | The `tilt_beta` used when `tilt_applied` is `true`, otherwise `null`. |
| `coord_frame`                 | String           | The `coord_frame` the request was read in: `"level"` or `"rig"`. |
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

##### Camera pose and tilt correction

The API looks at each label through a 90° perspective view (2048×2048 px) aimed at `x`/`y`, rendered by the
shared geometry package [sidewalk-panorama-tools](https://github.com/ProjectSidewalk/sidewalk-panorama-tools)
(`sidewalk_panorama_tools.geometry.label_view`), the same code the validator's training crops are made with.

A label placed by a person is stored in the viewer's gravity-levelled frame, but the panorama image is in the
camera rig's frame, which is tilted by the car's pitch and roll. So a stored point sits off its feature by about
`tilt_beta × (camera_pitch × cos b + camera_roll × sin b)` degrees of elevation, where `b` is the label's bearing in
the panorama. With `camera_pitch` given and `coord_frame=level`, the API rotates the view by that tilt, so the
feature the person clicked is at the centre of the view.

- **A request without `camera_pitch` gets no tilt correction**, whatever `camera_roll`, `tilt_beta` and
  `coord_frame` say. It reproduces the API 1.0.x view up to about one pixel at the label, two in the central half,
  from two geometry errors and one truncation in the old renderer that the shared package corrects (it read the
  source about one row low, its focal length was n/(n−1) too short, and it truncated pixel values instead of
  rounding them).
- `x = 1.0` is the same column as `x = 0.0` (the panorama wraps).
- `camera_pitch=0` with no (or zero) roll, or `tilt_beta=0`, is no rotation: `tilt_applied: false`, `tilt_beta: null`.
- A `camera_roll` without a `camera_pitch` is not applied: the API corrects only when the pitch is known. (The
  geometry package itself would apply a roll-only tilt, but no Street View label carries a roll today.)
- `coord_frame=rig` never corrects, even with `camera_pitch` given (`tilt_applied: false`).

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
| `400` | A required parameter is missing, `x`/`y` is outside 0–1, `label_type` isn't supported, or a pose field is invalid (`camera_pitch`/`camera_roll` not a number, infinite or beyond ±30, `tilt_beta` not a number, `NaN`, infinite or outside 0–1.5, or `coord_frame` not `level`/`rig`). |
| `401` | The API key is missing or wrong.                                                               |
| `500` | Model inference failed.                                                                        |
| `502` | The panorama couldn't be loaded from the cache or downloaded from Google (e.g. it no longer exists). |
