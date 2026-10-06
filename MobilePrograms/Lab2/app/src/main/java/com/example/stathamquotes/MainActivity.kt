package com.example.stathamquotes

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

val stathamQuotes = listOf(
    "Если ты не можешь решить проблему — значит, проблема в тебе.",
    "Я не ищу неприятности. Это они меня находят.",
    "Лучший план — это отсутствие плана. Тогда враг не знает, что ты задумал.",
    "Иногда, чтобы что-то найти, надо просто перестать искать.",
    "Не важно, сколько раз ты упал. Важно, сколько раз ты встал и пошёл дальше.",
    "Если тебя сбил грузовик — значит, ты недостаточно быстро бежал.",
    "Тишина — самый громкий ответ.",
    "Я не проигрываю. Я просто учусь, как не надо делать.",
    "Всё, что тебя не убило, ты просто недостаточно быстро добил.",
    "Слабые сдаются, сильные уходят за пивом. Я обычно иду за пивом."
)

@Composable
fun StathamQuoteScreen(modifier: Modifier = Modifier) {
    var currentQuote by remember { mutableStateOf("Нажми кнопку — и Стетхем заговорит.") }

    Column(
        modifier = modifier.fillMaxSize().padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Цитаты Джейсона Стетхема",
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(24.dp))

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f, fill = false)
                .verticalScroll(rememberScrollState()),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Text(
                text = "«$currentQuote»",
                modifier = Modifier.padding(20.dp),
                fontSize = 20.sp,
                fontStyle = FontStyle.Italic,
                textAlign = TextAlign.Center,
                lineHeight = 28.sp
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = { currentQuote = stathamQuotes.random() },
            modifier = Modifier.fillMaxWidth().height(56.dp)
        ) {
            Text(text = "Скажи что-нибудь, Джейсон", fontSize = 16.sp)
        }
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    StathamQuoteScreen()
                }
            }
        }
    }
}

@Preview(showBackground = true, showSystemUi = true)
@Composable
fun StathamQuoteScreenPreview() {
    MaterialTheme {
        StathamQuoteScreen()
    }
}