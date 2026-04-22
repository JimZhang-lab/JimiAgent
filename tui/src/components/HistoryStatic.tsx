import React from "react";
import { Static } from "ink";
import { useStore, type Message } from "../state/store.js";
import { MessageItem } from "./MessageItem.js";

/**
 * print-above 架构的历史打印器。
 *
 * 只把非流式消息交给 Ink `<Static>` 写进 scrollback。
 * 必须常驻在 App 根层，否则清会话后会重打历史。
 */
export function HistoryStatic(): React.ReactElement {
  const messages = useStore((s) => s.messages);

  // 只把非流式消息交给 Static；流式尾部留给 Messages 的 pending 区。
  const stable: Message[] = [];
  for (const m of messages) {
    if (!m.streaming) stable.push(m);
    else break; // 流式尾部及其后续都留给 pending 区
  }

  return (
    <Static items={stable}>
      {(m) => <MessageItem key={m.id} message={m} />}
    </Static>
  );
}
