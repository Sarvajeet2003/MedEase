import requests
import json

# FHIR server base URL
FHIR_SERVER_URL = "http://localhost:8080/fhir"  # Update this to match your FHIR server's local endpoint

# FHIR resources to be pushed
resources = [
    {
        "resourceType": "Patient",
        "id": "P1",
        "name": [{
            "use": "official",
            "family": "Doe",
            "given": ["John"]
        }],
        "gender": "other",
        "birthDate": "1955-01-01"
    },
    {
        "resourceType": "Observation",
        "id": "P1-a1c",
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "laboratory"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "4548-4",
                "display": "Hemoglobin A1c"
            }]
        },
        "subject": {
            "reference": "Patient/P1"
        },
        "valueCodeableConcept": {
            "text": "Abnormal"
        }
    },
    {
        "resourceType": "Encounter",
        "id": "P1-encounter",
        "status": "finished",
        "class": {
            "code": "emergency"
        },
        "subject": {
            "reference": "Patient/P1"
        },
        "reasonCode": [{
            "text": "Diagnosis code 1"
        }],
        "extension": [{
                "url": "http://example.org/fhir/structuredefinition/labProcedures",
                "valueInteger": 33
            },
            {
                "url": "http://example.org/fhir/structuredefinition/medications",
                "valueInteger": 2
            },
            {
                "url": "http://example.org/fhir/structuredefinition/outpatientVisits",
                "valueInteger": 4
            },
            {
                "url": "http://example.org/fhir/structuredefinition/inpatientVisits",
                "valueInteger": 1
            },
            {
                "url": "http://example.org/fhir/structuredefinition/emergencyVisits",
                "valueInteger": 1
            },
            {
                "url": "http://example.org/fhir/structuredefinition/diagnosesCount",
                "valueInteger": 5
            }
        ]
    },
    {
        "resourceType": "Patient",
        "id": "P2",
        "name": [{
            "use": "official",
            "family": "Smith",
            "given": ["Alice"]
        }],
        "gender": "female",
        "birthDate": "1992-01-01"
    },
    {
        "resourceType": "Observation",
        "id": "P2-a1c",
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "laboratory"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "4548-4",
                "display": "Hemoglobin A1c"
            }]
        },
        "subject": {
            "reference": "Patient/P2"
        },
        "valueCodeableConcept": {
            "text": "Abnormal"
        }
    },
    {
        "resourceType": "Encounter",
        "id": "P2-encounter",
        "status": "finished",
        "class": {
            "code": "inpatient"
        },
        "subject": {
            "reference": "Patient/P2"
        },
        "reasonCode": [{
            "text": "Diagnosis code 0"
        }],
        "extension": [{
                "url": "http://example.org/fhir/structuredefinition/labProcedures",
                "valueInteger": 81
            },
            {
                "url": "http://example.org/fhir/structuredefinition/medications",
                "valueInteger": 10
            },
            {
                "url": "http://example.org/fhir/structuredefinition/outpatientVisits",
                "valueInteger": 4
            },
            {
                "url": "http://example.org/fhir/structuredefinition/inpatientVisits",
                "valueInteger": 4
            },
            {
                "url": "http://example.org/fhir/structuredefinition/emergencyVisits",
                "valueInteger": 1
            },
            {
                "url": "http://example.org/fhir/structuredefinition/diagnosesCount",
                "valueInteger": 6
            }
        ]
    },
    {
        "resourceType": "Patient",
        "id": "P3",
        "name": [{
            "use": "official",
            "family": "Brown",
            "given": ["Ethan"]
        }],
        "gender": "male",
        "birthDate": "1946-01-01"
    },
    {
        "resourceType": "Observation",
        "id": "P3-a1c",
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "laboratory"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "4548-4",
                "display": "Hemoglobin A1c"
            }]
        },
        "subject": {
            "reference": "Patient/P3"
        },
        "valueCodeableConcept": {
            "text": "Normal"
        }
    },
    {
        "resourceType": "Encounter",
        "id": "P3-encounter",
        "status": "finished",
        "class": {
            "code": "inpatient"
        },
        "subject": {
            "reference": "Patient/P3"
        },
        "reasonCode": [{
            "text": "Diagnosis code 1"
        }],
        "extension": [{
                "url": "http://example.org/fhir/structuredefinition/labProcedures",
                "valueInteger": 75
            },
            {
                "url": "http://example.org/fhir/structuredefinition/medications",
                "valueInteger": 29
            },
            {
                "url": "http://example.org/fhir/structuredefinition/outpatientVisits",
                "valueInteger": 4
            },
            {
                "url": "http://example.org/fhir/structuredefinition/inpatientVisits",
                "valueInteger": 0
            },
            {
                "url": "http://example.org/fhir/structuredefinition/emergencyVisits",
                "valueInteger": 3
            },
            {
                "url": "http://example.org/fhir/structuredefinition/diagnosesCount",
                "valueInteger": 5
            }
        ]
    }
]

def push_to_fhir_server(resource, fhir_server_url):
    """
    Pushes a single FHIR resource to the FHIR server.

    :param resource: The FHIR resource to push.
    :param fhir_server_url: The base URL of the FHIR server.
    :return: The response from the server.
    """
    url = f"{fhir_server_url}/{resource['resourceType']}"
    if 'id' in resource:  # Use the ID for updates
        url = f"{url}/{resource['id']}"
    headers = {"Content-Type": "application/fhir+json"}
    response = requests.put(url, headers=headers, data=json.dumps(resource))
    return response

# Sort resources so that Patient resources are uploaded first
resources.sort(key=lambda x: x["resourceType"] != "Patient")

# Push resources to the FHIR server
for resource in resources:
    response = push_to_fhir_server(resource, FHIR_SERVER_URL)
    if response.status_code in [200, 201]:
        print(f"Successfully uploaded resource {resource['id']} of type {resource['resourceType']}.")
    else:
        print(f"Failed to upload resource {resource['id']} of type {resource['resourceType']}.")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
