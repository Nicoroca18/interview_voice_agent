
import sys
import argparse
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from src.agent import InterviewAgent
from src.voice_handler import VoiceHandler
from src.config import AGENT_NAME, COMPANY_NAME, POSITION

console = Console()


def print_voice_header():
    """Print voice mode header."""
    console.print("\n")
    console.print(Panel.fit(
        f"[bold cyan]{AGENT_NAME}[/bold cyan]\n"
        f"[yellow]🎤 Voice Interview Mode[/yellow]\n"
        f"Position: [green]{POSITION}[/green] at [green]{COMPANY_NAME}[/green]",
        border_style="cyan"
    ))
    console.print("\n")


def run_voice_interview(
    api_key: Optional[str] = None,
    tts_engine: str = "gtts",
    whisper_model: str = "base",
    test_audio: bool = True
):
    """
    Run a voice-based interview session.
    
    Args:
        api_key: Anthropic API key
        tts_engine: TTS engine ('gtts' or 'openai')
        whisper_model: Whisper model size
        test_audio: Whether to test audio before starting
    """
    print_voice_header()
    
    try:
        # Initialize voice handler
        console.print("[yellow]Initializing voice system...[/yellow]")
        voice_handler = VoiceHandler(
            tts_engine=tts_engine,
            whisper_model=whisper_model
        )
        console.print("[green]✓ Voice system ready![/green]\n")
        
        # Test audio if requested
        if test_audio:
            console.print("[bold]Testing audio setup...[/bold]")
            
            # Test microphone
            console.print("\n[cyan]Microphone Test[/cyan]")
            console.print("Say something for 3 seconds...")
            
            if not voice_handler.test_microphone():
                console.print("[red]Microphone test failed. Please check your microphone.[/red]")
                response = input("\nContinue anyway? (y/n): ")
                if response.lower() != 'y':
                    return
            
            # Test speakers
            console.print("\n[cyan]Speaker Test[/cyan]")
            if not voice_handler.test_speakers():
                console.print("[yellow]Please check your speakers/headphones.[/yellow]")
                response = input("\nContinue anyway? (y/n): ")
                if response.lower() != 'y':
                    return
            
            console.print("\n[green]✓ Audio setup complete![/green]\n")
        
        # Initialize interview agent
        console.print("[yellow]Initializing interview agent...[/yellow]")
        agent = InterviewAgent(api_key=api_key)
        console.print("[green]✓ Agent ready![/green]\n")
        
        # Start interview
        console.print("[bold cyan]Starting voice interview...[/bold cyan]\n")
        console.print("[dim]Press Ctrl+C to end interview at any time[/dim]\n")
        
        opening = agent.start_interview()
        voice_handler.speak(opening, language='en')
        
        # Main conversation loop
        while not agent.is_interview_complete:
            try:
                # Listen to user
                console.print("\n[bold green]Your turn to speak...[/bold green]")
                console.print("[dim](Speak naturally, then pause. The system will detect when you're done)[/dim]")
                
                user_input = voice_handler.listen_and_transcribe()
                
                # Skip empty inputs
                if not user_input or len(user_input.strip()) < 3:
                    console.print("[yellow]No speech detected. Please try again.[/yellow]")
                    continue
                
                console.print(f"[green]You said:[/green] {user_input}\n")
                
                # Check for exit commands - comprehensive list
                exit_phrases = [
                    'end interview', 'end the interview', 'finish interview', 'finish the interview',
                    'stop interview', 'stop the interview', 'goodbye', 'good bye', 'bye bye',
                    'bye-bye', 'bye', 'adiós', 'adios', 'see you', 'see you soon',
                    'that\'s all', 'we\'re done', 'i\'m done', 'exit', 'quit',
                    'terminate', 'close', 'thanks bye', 'thank you bye', 'hasta luego',
                    'chao', 'ciao', 'au revoir', 'auf wiedersehen'
                ]
                
                if any(phrase in user_input.lower() for phrase in exit_phrases):
                    console.print("\n[yellow]Interview ended by candidate.[/yellow]")
                    break
                
                # Process with agent
                response = agent.process_message(user_input)
                
                # Check if interview completed after this message
                if agent.is_interview_complete:
                    console.print("\n[yellow]Interview has been completed automatically.[/yellow]")
                    break
                
                # Speak response only if interview is not complete
                voice_handler.speak(response, language='en')
                
                # Show status
                status = agent.get_interview_status()
                console.print(f"[dim]Turns: {status['total_turns']}, Duration: {status['duration']}[/dim]\n")
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Interview interrupted by user.[/yellow]")
                
                response = input("\nDo you want to end the interview? (y/n): ")
                if response.lower() == 'y':
                    break
                else:
                    console.print("[cyan]Resuming interview...[/cyan]\n")
                    continue
            
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                console.print("[yellow]Please try again.[/yellow]\n")
                continue
        
        # End interview and generate report
        console.print("\n[yellow]Generating interview report...[/yellow]\n")
        
        try:
            results = agent.end_interview()
            
            # Display results
            console.print(Panel.fit(
                "[bold green]Interview Complete![/bold green]",
                border_style="green"
            ))
            
            console.print(f"\n[cyan]Interview ID:[/cyan] {results['interview_id']}")
            console.print(f"[cyan]Duration:[/cyan] {results['duration_estimate']}")
            console.print(f"[cyan]Total Turns:[/cyan] {results['total_turns']}")
            
            # Display candidate info
            console.print("\n[bold]Candidate Information:[/bold]")
            candidate = results['candidate_info']
            
            if candidate.get('candidate_name'):
                console.print(f"  Name: {candidate['candidate_name']}")
            if candidate.get('email'):
                console.print(f"  Email: {candidate['email']}")
            if candidate.get('years_of_experience') is not None:
                console.print(f"  Experience: {candidate['years_of_experience']} years")
            
            # Display assessment
            console.print("\n[bold]Assessment:[/bold]")
            assessment = results['assessment']
            
            if assessment.get('overall_sentiment'):
                console.print(f"  Sentiment: {assessment['overall_sentiment']}")
            if assessment.get('recommendation'):
                console.print(f"  Recommendation: {assessment['recommendation']}")
            
            # Display summary
            console.print("\n[bold]Summary:[/bold]")
            console.print(Panel(results['conversation_summary'], border_style="blue"))
            
            console.print(f"\n[green]✓ Report saved to data/conversations/{results['interview_id']}_complete.json[/green]")
            
        except Exception as e:
            console.print(f"[red]Error generating report: {str(e)}[/red]")
        
        finally:
            # Cleanup
            voice_handler.cleanup()
    
    except Exception as e:
        console.print(f"\n[red]Error initializing voice interview: {str(e)}[/red]")
        console.print("[yellow]Please check your audio devices and dependencies.[/yellow]")
        sys.exit(1)
    
    console.print("\n[cyan]Thank you for using Voice Interview Mode![/cyan]\n")


def test_voice_only():
    """Test voice system only, without interview."""
    console.print("\n[bold cyan]Voice System Test[/bold cyan]\n")
    
    try:
        console.print("[yellow]Initializing...[/yellow]")
        handler = VoiceHandler(tts_engine="gtts", whisper_model="base")
        
        # Test microphone
        console.print("\n[bold]1. Testing Microphone[/bold]")
        handler.test_microphone()
        
        # Test speakers
        console.print("\n[bold]2. Testing Speakers[/bold]")
        handler.test_speakers()
        
        # Interactive test
        console.print("\n[bold]3. Interactive Test[/bold]")
        console.print("I will listen to you and repeat what you said...\n")
        
        text = handler.listen_and_transcribe()
        
        if text:
            handler.speak(f"You said: {text}")
        
        handler.cleanup()
        console.print("\n[green]✓ Voice test complete![/green]\n")
        
    except Exception as e:
        console.print(f"\n[red]Test failed: {str(e)}[/red]\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Voice Interview Mode - AI-powered voice interviews"
    )
    
    parser.add_argument(
        "--tts",
        choices=["gtts", "openai"],
        default="gtts",
        help="TTS engine to use (default: gtts - free Google TTS)"
    )
    
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)"
    )
    
    parser.add_argument(
        "--no-audio-test",
        action="store_true",
        help="Skip audio testing before interview"
    )
    
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test voice system, don't run interview"
    )
    
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    if args.test_only:
        test_voice_only()
        return
    
    # Display instructions
    console.print("\n[bold cyan]Voice Interview Mode[/bold cyan]\n")
    console.print("[bold]How it works:[/bold]")
    console.print("1. The AI interviewer will ask you questions (spoken)")
    console.print("2. You speak your answers naturally")
    console.print("3. The system automatically detects when you're done speaking")
    console.print("4. Say 'end interview' or 'goodbye' to finish early\n")
    
    console.print("[yellow]Requirements:[/yellow]")
    console.print("- Working microphone")
    console.print("- Working speakers/headphones")
    console.print("- Quiet environment (for best results)")
    console.print("- Anthropic API key set in .env or --api-key\n")
    
    # Check API key
    if not args.api_key:
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            console.print("[red]Error: No API key found[/red]")
            console.print("Set ANTHROPIC_API_KEY in .env or use --api-key")
            sys.exit(1)
    
    # Start interview
    try:
        run_voice_interview(
            api_key=args.api_key,
            tts_engine=args.tts,
            whisper_model=args.whisper_model,
            test_audio=not args.no_audio_test
        )
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Exiting...[/yellow]\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
