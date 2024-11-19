import torch  # For PyTorch operations
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # For loading pre-trained models and tokenizers

# Utility function to divide a list into chunks
def chunks(lst, n):
    """
    Yield successive n-sized chunks from a list.
    Args:
        lst (list): List to be chunked.
        n (int): Size of each chunk.
    Yields:
        list: Chunks of the original list.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]  # Yield slices of the list with size `n`.

# Utility function to trim unnecessary padding tokens from a batch
def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """
    Remove columns that are exclusively populated with pad_token_id.
    Args:
        input_ids (Tensor): Input IDs of shape `(batch_size, seq_len)`.
        pad_token_id (int): ID representing padding tokens.
        attention_mask (Tensor, optional): Attention mask corresponding to input_ids.
    Returns:
        Tensor or Tuple[Tensor, Tensor]: Trimmed input IDs (and attention mask, if provided).
    """
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)  # Identify columns with non-padding tokens.
    if attention_mask is None:
        return input_ids[:, keep_column_mask]  # Return only the trimmed input IDs.
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])  # Trim both tensors.

# Utility function to apply task-specific parameters
def use_task_specific_params(model, task):
    """
    Update model configuration with task-specific parameters, if available.
    Args:
        model (PreTrainedModel): The pre-trained model to update.
        task (str): Task name to fetch parameters for (e.g., "summarization").
    """
    task_specific_params = model.config.task_specific_params  # Fetch task-specific parameters from config.

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})  # Get parameters specific to the task.
        model.config.update(pars)  # Update the model configuration with these parameters.

# Class for generating commonsense knowledge using a Comet model
class Comet:
    def __init__(self, model_path, device):
        """
        Initialize the Comet model for commonsense knowledge generation.
        Args:
            model_path (str): Path to the pre-trained model.
            device (torch.device): Device to load the model on (e.g., "cuda" or "cpu").
        """
        # Load the pre-trained model and tokenizer from the given path
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device  # Set the device (e.g., CPU or GPU).
        self.batch_size = 1  # Default batch size for generation.

        # Initialize model parameters for the task
        use_task_specific_params(self.model, "summarization")
        self.model.zero_grad()  # Reset gradients to ensure clean state.

    def generate(self, input_event, rel):
        """
        Generate commonsense knowledge for a given input event and relation.
        Args:
            input_event (str): Input text event (e.g., "Person X eats dinner").
            rel (str): Relationship type (e.g., "xWant", "xEffect").
        Returns:
            list: List of generated commonsense statements for the specified relation.
        """
        # Construct the query with the input event and relation
        query = "{} {} [GEN]".format(input_event, rel)

        with torch.no_grad():  # Disable gradient computation for faster inference
            # Tokenize the query and convert it into a tensor
            query = self.tokenizer(
                query, return_tensors="pt", truncation=True, padding="max_length"
            ).to(self.device)

            # Trim padding tokens from the input batch
            input_ids, attention_mask = trim_batch(
                **query, pad_token_id=self.tokenizer.pad_token_id
            )

            # Generate output sequences using beam search
            summaries = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_start_token_id=None,  # Start decoding from the default token
                num_beams=5,  # Use 5 beams for beam search
                num_return_sequences=5,  # Generate 5 sequences for each input
            )

            # Decode the generated sequences into readable text
            dec = self.tokenizer.batch_decode(
                summaries,
                skip_special_tokens=True,  # Skip special tokens like [PAD], [SEP]
                clean_up_tokenization_spaces=False,  # Preserve tokenization spaces
            )

            return dec  # Return the list of generated sequences.
