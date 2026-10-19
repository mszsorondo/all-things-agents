"""
XES Converter for Process Mining
Converts event logs to XES format and PM4Py-compatible dataframes
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

class XESConverter:
    """
    Converts event logs to XES format for use with ProM and other process mining tools
    """

    def __init__(self, event_log: Dict[str, Any]):
        """
        Initialize with an event log dictionary

        Args:
            event_log: Dictionary containing the event log from ProcessMiningLogger
        """
        self.event_log = event_log
        self.case_id = event_log.get("case_id", "unknown")
        self.events = event_log.get("events", [])

    def to_xes_xml(self, output_path: Optional[Path] = None) -> str:
        """
        Convert event log to XES XML format

        Args:
            output_path: Optional path to save the XES file

        Returns:
            XES XML string
        """
        # Create root element
        root = ET.Element("log", {
            "xes.version": "1.0",
            "xes.features": "nested-attributes",
            "xmlns": "http://www.xes-standard.org/"
        })

        # Add extensions
        extensions = [
            ("concept", "http://www.xes-standard.org/concept.xesext"),
            ("time", "http://www.xes-standard.org/time.xesext"),
            ("org", "http://www.xes-standard.org/org.xesext"),
            ("lifecycle", "http://www.xes-standard.org/lifecycle.xesext")
        ]

        for name, uri in extensions:
            ET.SubElement(root, "extension", {
                "name": name,
                "prefix": name,
                "uri": uri
            })

        # Add global attributes
        global_trace = ET.SubElement(root, "global", {"scope": "trace"})
        ET.SubElement(global_trace, "string", {
            "key": "concept:name",
            "value": "__INVALID__"
        })

        global_event = ET.SubElement(root, "global", {"scope": "event"})
        for key in ["concept:name", "org:resource", "lifecycle:transition"]:
            ET.SubElement(global_event, "string", {
                "key": key,
                "value": "__INVALID__"
            })
        ET.SubElement(global_event, "date", {
            "key": "time:timestamp",
            "value": "1970-01-01T00:00:00.000+00:00"
        })

        # Create trace
        trace = ET.SubElement(root, "trace")

        # Add trace attributes
        ET.SubElement(trace, "string", {
            "key": "concept:name",
            "value": self.case_id
        })
        ET.SubElement(trace, "string", {
            "key": "architecture",
            "value": self.event_log.get("architecture", "unknown")
        })
        ET.SubElement(trace, "string", {
            "key": "problem",
            "value": self.event_log.get("problem", "unknown")
        })

        # Add events
        for event_dict in self.events:
            event = ET.SubElement(trace, "event")

            # Standard attributes
            ET.SubElement(event, "string", {
                "key": "concept:name",
                "value": event_dict.get("activity", "unknown")
            })
            ET.SubElement(event, "string", {
                "key": "org:resource",
                "value": event_dict.get("resource", "unknown")
            })
            ET.SubElement(event, "string", {
                "key": "lifecycle:transition",
                "value": event_dict.get("lifecycle", "complete")
            })
            ET.SubElement(event, "date", {
                "key": "time:timestamp",
                "value": event_dict.get("timestamp", datetime.utcnow().isoformat() + "Z")
            })

            # Custom attributes
            attributes = event_dict.get("attributes", {})
            for key, value in attributes.items():
                if value is not None:
                    if isinstance(value, (int, float)):
                        ET.SubElement(event, "int" if isinstance(value, int) else "float", {
                            "key": f"custom:{key}",
                            "value": str(value)
                        })
                    elif isinstance(value, bool):
                        ET.SubElement(event, "boolean", {
                            "key": f"custom:{key}",
                            "value": str(value).lower()
                        })
                    elif isinstance(value, list):
                        ET.SubElement(event, "string", {
                            "key": f"custom:{key}",
                            "value": json.dumps(value)
                        })
                    else:
                        ET.SubElement(event, "string", {
                            "key": f"custom:{key}",
                            "value": str(value)
                        })

        # Convert to pretty XML string
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")

        # Save if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)

        return pretty_xml

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert event log to pandas DataFrame for PM4Py

        Returns:
            DataFrame with columns suitable for PM4Py
        """
        rows = []

        for event in self.events:
            row = {
                'case:concept:name': self.case_id,
                'concept:name': event.get('activity', 'unknown'),
                'time:timestamp': pd.to_datetime(event.get('timestamp', datetime.utcnow().isoformat())),
                'org:resource': event.get('resource', 'unknown'),
                'lifecycle:transition': event.get('lifecycle', 'complete')
            }

            # Add custom attributes
            attributes = event.get('attributes', {})
            for key, value in attributes.items():
                if value is not None:
                    # Handle lists and dicts
                    if isinstance(value, (list, dict)):
                        row[f'custom:{key}'] = json.dumps(value)
                    else:
                        row[f'custom:{key}'] = value

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by timestamp
        df = df.sort_values('time:timestamp')

        return df

    def to_csv(self, output_path: Path):
        """
        Export event log to CSV format

        Args:
            output_path: Path to save the CSV file
        """
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)

def to_pm4py_dataframe(event_log_path: Path) -> pd.DataFrame:
    """
    Load an event log from JSON and convert to PM4Py dataframe

    Args:
        event_log_path: Path to the event_log.json file

    Returns:
        DataFrame ready for PM4Py processing
    """
    with open(event_log_path, 'r') as f:
        event_log = json.load(f)

    converter = XESConverter(event_log)
    return converter.to_dataframe()

def batch_convert_to_xes(experiment_dir: Path, output_dir: Optional[Path] = None):
    """
    Convert all event logs in an experiments directory to XES format

    Args:
        experiment_dir: Directory containing experiment runs
        output_dir: Optional directory for XES files (defaults to experiment_dir)
    """
    if output_dir is None:
        output_dir = experiment_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all event_log.json files
    event_logs = list(Path(experiment_dir).glob("*/event_log.json"))

    print(f"Found {len(event_logs)} event logs to convert")

    for log_path in event_logs:
        run_name = log_path.parent.name

        # Load event log
        with open(log_path, 'r') as f:
            event_log = json.load(f)

        # Convert to XES
        converter = XESConverter(event_log)

        # Save XES file
        xes_path = output_dir / f"{run_name}.xes"
        converter.to_xes_xml(xes_path)

        # Save CSV file
        csv_path = output_dir / f"{run_name}.csv"
        converter.to_csv(csv_path)

        print(f"✅ Converted {run_name}: XES and CSV saved")

    print(f"📊 All conversions complete. Files saved to {output_dir}")

def merge_event_logs(event_logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Merge multiple event logs into a single dataframe for comparative analysis

    Args:
        event_logs: List of event log dictionaries

    Returns:
        Combined dataframe with all cases
    """
    all_dfs = []

    for event_log in event_logs:
        converter = XESConverter(event_log)
        df = converter.to_dataframe()

        # Add metadata columns
        df['architecture'] = event_log.get('architecture', 'unknown')
        df['problem'] = event_log.get('problem', 'unknown')

        all_dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Sort by case and timestamp
    combined_df = combined_df.sort_values(['case:concept:name', 'time:timestamp'])

    return combined_df