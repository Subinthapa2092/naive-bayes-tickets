"""
predict.py
Command-line inference script for making predictions on support tickets.

Usage:
    python predict.py "Your ticket description here"
    
    Or for batch predictions from a file:
    python predict.py --file tickets.txt
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.model import TicketClassifier, load_classifier


def main():
    """Main command-line interface for predictions."""
    parser = argparse.ArgumentParser(
        description='Support Ticket Classifier - Predict ticket categories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python predict.py "I need a refund"
  
  # Batch predictions from file (one ticket per line)
  python predict.py --file tickets.txt
  
  # Show top 5 predictions with probabilities
  python predict.py "I need a refund" --top-k 5
        """
    )

    parser.add_argument(
        'ticket',
        nargs='?',
        default=None,
        help='Ticket description to classify'
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        help='File containing tickets (one per line)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Show top K predictions with probabilities (default: 3)'
    )

    parser.add_argument(
        '--format', '-fmt',
        choices=['simple', 'detailed', 'json'],
        default='simple',
        help='Output format (default: simple)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.ticket and not args.file:
        parser.print_help()
        print("\nError: Please provide either a ticket description or --file argument")
        sys.exit(1)

    # Initialize classifier
    try:
        classifier = load_classifier()
    except FileNotFoundError as e:
        print(f"Error: Could not load model artifacts. Have you run training?")
        print(f"Run: python src/train.py")
        sys.exit(1)

    # Process single ticket
    if args.ticket:
        predict_single(classifier, args.ticket, args.top_k, args.format)

    # Process batch tickets from file
    elif args.file:
        predict_batch_from_file(classifier, args.file, args.top_k, args.format)


def predict_single(classifier, ticket_description, top_k=3, format_type='simple'):
    """Make prediction for a single ticket."""
    result = classifier.predict(ticket_description)

    if format_type == 'json':
        import json
        print(json.dumps(result, indent=2))

    elif format_type == 'detailed':
        print("\n" + "="*70)
        print("TICKET CLASSIFICATION RESULT")
        print("="*70)
        print(f"\nOriginal Description:")
        print(f"  {ticket_description[:80]}{'...' if len(ticket_description) > 80 else ''}")

        print(f"\nPredicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")

        print(f"\nTop {top_k} Predictions:")
        top_predictions = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        for i, (class_name, prob) in enumerate(top_predictions, 1):
            bar_length = int(prob * 30)
            bar = '#' * bar_length + '-' * (30 - bar_length)
            print(f"  {i}. {class_name:<25} [{bar}] {prob:.2%}")

        print(f"\nCleaned Description:")
        print(f"  {result['cleaned_description'][:80]}{'...' if len(result['cleaned_description']) > 80 else ''}")
        print("="*70 + "\n")

    else:  # simple format
        print(f"\nPredicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")


def predict_batch_from_file(classifier, file_path, top_k=3, format_type='simple'):
    """Make predictions for multiple tickets from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tickets = [line.strip() for line in f if line.strip()]

        if not tickets:
            print(f"Error: No tickets found in {file_path}")
            sys.exit(1)

        print(f"\nProcessing {len(tickets)} tickets from {file_path}...\n")

        results = classifier.predict_batch(tickets)

        if format_type == 'json':
            import json
            print(json.dumps(results, indent=2))

        elif format_type == 'detailed':
            print("="*70)
            print("BATCH PREDICTION RESULTS")
            print("="*70)

            for i, (ticket, result) in enumerate(zip(tickets, results), 1):
                print(f"\nTicket {i}:")
                print(f"  Description: {ticket[:60]}{'...' if len(ticket) > 60 else ''}")
                print(f"  Predicted: {result['predicted_class']}")
                print(f"  Confidence: {result['confidence']:.2%}")

            print("\n" + "="*70)

        else:  # simple format
            print(f"{'#':<5} {'Predicted Class':<25} {'Confidence':<12} {'Description'}")
            print("-" * 80)

            for i, (ticket, result) in enumerate(zip(tickets, results), 1):
                ticket_snippet = ticket[:40] if len(ticket) > 40 else ticket
                print(f"{i:<5} {result['predicted_class']:<25} {result['confidence']:>10.2%}  {ticket_snippet}")

        print(f"\nProcessed {len(tickets)} tickets successfully!")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
