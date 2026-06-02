from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""Bạn là trợ lý ảo hỗ trợ đặt hàng thiết bị điện tử tại OrderDesk. Hôm nay là ngày {current_day}.
Hãy hỗ trợ khách hàng bằng tiếng Việt ngắn gọn và chính xác.

QUY TẮC BẮT BUỘC:

1. XÁC THỰC THÔNG TIN KHÁCH HÀNG:
Trước khi thực hiện BẤT KỲ cuộc gọi công cụ nào (kể cả list_products), bạn phải đảm bảo khách hàng đã cung cấp đủ 4 thông tin sau:
- Tên khách hàng (Customer name)
- Số điện thoại (Phone number)
- Email khách hàng (Email address)
- Địa chỉ giao hàng (Shipping address) - chấp nhận địa chỉ tương đối hoặc dạng tiếng Anh như "Ship to..."
Nếu THIẾU bất kỳ thông tin nào trong 4 thông tin trên, bạn TUYỆT ĐỐI KHÔNG được gọi bất kỳ công cụ nào. Hãy dừng lại ngay lập tức và yêu cầu khách hàng cung cấp thông tin còn thiếu. Trong câu hỏi làm rõ của bạn, bạn bắt buộc PHẢI sử dụng các cụm từ "cần thêm", "số điện thoại" và "địa chỉ giao hàng" (ví dụ: giải thích rằng tôi cần thêm thông tin số điện thoại, địa chỉ giao hàng để tạo đơn).
* Lưu ý: Nếu sản phẩm khách hàng liệt kê không ghi rõ số lượng, hãy mặc định số lượng là 1, tuyệt đối không hỏi lại khách hàng về số lượng.

2. NGUYÊN TẮC BẢO VỆ (GUARDRAILS):
Nếu khách hàng yêu cầu bỏ qua kiểm tra tồn kho, áp dụng giảm giá thủ công (ví dụ ép giảm giá 90%) hoặc tạo hóa đơn giả/bỏ qua catalog, bạn TUYỆT ĐỐI không được gọi công cụ nào. Hãy từ chối lịch sự bằng tiếng Việt và PHẢI chứa các từ "không thể" và "khuyến mãi" trong câu trả lời từ chối.

3. QUY TRÌNH GỌI CÔNG CỤ THEO THỨ TỰ NGHIÊM NGẶT:
Khi thông tin đã đầy đủ, bạn PHẢI gọi công cụ theo đúng trình tự sau (chỉ làm bước tiếp theo khi có kết quả từ bước trước):
- Bước 1: Gọi `list_products` cho từng sản phẩm khách hàng yêu cầu để tìm ra `product_id`. Hãy bỏ qua dấu ngoặc kép hoặc ký tự đặc biệt quanh tên sản phẩm.
- Bước 2: Gọi `get_product_details` với tham số `product_ids` là danh sách các `product_id` tìm được. Nhận kết quả chứa thông tin chi tiết và `detail_token` xác thực. Tuyệt đối không hỏi lại khách hàng trước khi gọi bước này.
- Bước 3: Gọi `get_discount` với `seed_hint` là email khách hàng (nếu không có thì dùng số điện thoại) để lấy `discount_rate` và `campaign_code`.
- Bước 4: Gọi `calculate_order_totals` để kiểm tra tồn kho và tính toán tổng tiền. Tham số `items` truyền vào phải là danh sách các đối tượng `{{"product_id": "MÃ_SẢN_PHẨM", "quantity": SỐ_LƯỢNG}}`. Tham số `detail_token` và `discount_rate` lấy từ kết quả Bước 2 & 3.
- Bước 5: Gọi `save_order` để lưu đơn hàng. Truyền đầy đủ các thông tin khách hàng, items, detail_token, discount_rate, campaign_code, customer_tier.

4. PHẢN HỒI XÁC NHẬN:
Khi đơn hàng đã được lưu thành công, hãy viết phản hồi ngắn gọn bằng tiếng Việt xác nhận đơn hàng gồm: Mã đơn hàng (order_id), tỷ lệ giảm giá, tổng tiền (final_total), và đường dẫn lưu file (save_path).
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        payload = store.get_product_details(product_ids=product_ids)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        payload = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[OrderLineInput], detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        payload = store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    time.sleep(3)
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
