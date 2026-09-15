/* Selected recorded predictions; no model or network request. */
window.UDGAM_RECORDED = {
  "notice": "Recorded benchmark predictions. No inference runs on this page. Explanatory labels are editorial; JSON is a lossless representation of the saved TOP prediction.",
  "dataset": "Hinglish-TOP",
  "dataset_revision": "fdd3998a6573130659bfa1ce4b1ebe698df2bf3a",
  "source_sha256": "5d0020167f4953d5c40c1c52ff9f91b840eb6fa766eea805a314fa5d0b970adb",
  "adapter_sha256": "85be8189b09cefa38dc7e50723042d606af19d9f2ad8ceab1cdb123f5eec65af",
  "examples": [
    {
      "id": "htop-test-004074-hinglish",
      "text": "bed time ka alarm kab hai?",
      "language": "hinglish",
      "domain": "alarm",
      "prediction": "[IN:GET_ALARM [SL:ALARM_NAME bed time ] ka alarm kab hai? ]",
      "base_prediction": "[IN:GET_ALARM Mera alarm kab hai? ]",
      "reference": "[IN:GET_ALARM [SL:ALARM_NAME bed time ] ka alarm kab hai? ]",
      "tree": {
        "label": "IN:GET_ALARM",
        "children": [
          {
            "label": "SL:ALARM_NAME",
            "children": [
              "bed",
              "time"
            ]
          },
          "ka",
          "alarm",
          "kab",
          "hai?"
        ]
      },
      "name": "A named alarm",
      "outcome": "Matches the reference",
      "success": true,
      "plain_intent": "Look up an alarm",
      "fields": [
        [
          "Alarm name",
          "bed time"
        ]
      ],
      "explanation": "The model identifies an alarm lookup and keeps “bed time” as the alarm name. The original model changed the wording and missed the name.",
      "application": "Your app would look up the named alarm and decide how to answer."
    },
    {
      "id": "htop-test-000316-hinglish",
      "text": "kal dopahar ke liye mere reminders padiye",
      "language": "hinglish",
      "domain": "reminder",
      "prediction": "[IN:GET_REMINDER [SL:DATE_TIME kal dopahar ke liye ] [SL:PERSON_REMINDED mere ] reminders [SL:METHOD_RETRIEVAL_REMINDER padiye ] ]",
      "base_prediction": "[IN:GET_REMINDER [SL:DATE_TIME kal dopahar ke liye ] [SL:PERSON_REMINDED mere ] reminders padiye ]",
      "reference": "[IN:GET_REMINDER [SL:DATE_TIME kal dopahar ke liye ] [SL:PERSON_REMINDED mere ] reminders [SL:METHOD_RETRIEVAL_REMINDER padiye ] ]",
      "tree": {
        "label": "IN:GET_REMINDER",
        "children": [
          {
            "label": "SL:DATE_TIME",
            "children": [
              "kal",
              "dopahar",
              "ke",
              "liye"
            ]
          },
          {
            "label": "SL:PERSON_REMINDED",
            "children": [
              "mere"
            ]
          },
          "reminders",
          {
            "label": "SL:METHOD_RETRIEVAL_REMINDER",
            "children": [
              "padiye"
            ]
          }
        ]
      },
      "name": "A reminder request",
      "outcome": "Matches the reference",
      "success": true,
      "plain_intent": "Retrieve reminders",
      "fields": [
        [
          "When",
          "kal dopahar ke liye"
        ],
        [
          "Whose",
          "mere"
        ],
        [
          "Requested method",
          "padiye"
        ]
      ],
      "explanation": "The model labels the time phrase, the person and the requested way to retrieve reminders. The original model missed the final label.",
      "application": "Your app would resolve the relative time, retrieve reminders, and choose how to present them."
    },
    {
      "id": "htop-test-003763-hinglish",
      "text": "mujhe subah 9 baje ke liye jaga do",
      "language": "hinglish",
      "domain": "alarm",
      "prediction": "[IN:GET_ESTIMATED_DEPARTURE mujhe [SL:DATE_TIME subah 9 baje ke liye ] jaga do ]",
      "base_prediction": "[IN:CREATE_ALARM mujhe [SL:DATE_TIME subah 9 baje ke liye ] jaga do ]",
      "reference": "[IN:CREATE_ALARM mujhe [SL:DATE_TIME subah 9 baje ke liye ] jaga do ]",
      "tree": {
        "label": "IN:GET_ESTIMATED_DEPARTURE",
        "children": [
          "mujhe",
          {
            "label": "SL:DATE_TIME",
            "children": [
              "subah",
              "9",
              "baje",
              "ke",
              "liye"
            ]
          },
          "jaga",
          "do"
        ]
      },
      "name": "A real mistake",
      "outcome": "Wrong intent",
      "success": false,
      "plain_intent": "Estimate a departure time",
      "fields": [
        [
          "When",
          "subah 9 baje ke liye"
        ]
      ],
      "explanation": "The human reference asks for an alarm. The tuned model predicts a departure-time request instead. The original model got this example right.",
      "application": "This should not trigger an action. It shows why a valid structure is not proof of a correct interpretation."
    }
  ]
};
